import faiss
import numpy as np

from utils import *


class FaissNNeighbors:
    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.index = None
        self.title_index = None
        self.body_index = None
        self.title_body_index = None
        self.model = SentenceTransformer(model)

    def model_encode(self, sentences):
        print(sentences)
        return self.model.encode(sentences)

    def get_embeddings(self, records):
        title, body = [], []
        title_body = []
        for i, record in enumerate(records):
            title.append(record['TITLE_RAW'])
            body.append(record['BODY'])
            title_body.append(f"{record['TITLE_RAW']} : {record['BODY']}")
        return self.model_encode(title), self.model_encode(body), self.model_encode(title_body)

    def create_index(self, X_train):
        index = faiss.IndexIDMap(faiss.IndexFlatIP(X_train.shape[1]))
        faiss.normalize_L2(X_train)
        index.add_with_ids(X_train, np.arange(X_train.shape[0]))
        return index

    def fit(self, records):
        title_embeddings, body_embeddings = [], []
        title_body_embeddings = []
        for i, record in enumerate(records):
            title_embeddings.append(record['title_embedding'])
            body_embeddings.append(record['body_embedding'])
            title_body_embeddings.append(record['title_body_embedding'])

        title_train = np.array(title_embeddings)
        body_train = np.array(body_embeddings)
        title_body_train = np.array(title_body_embeddings)

        self.title_index = self.create_index(title_train)
        self.body_index = self.create_index(body_train)
        self.title_body_index = self.create_index(title_body_train)


    def search_index(self, record, topk=5):


        title_embedding = record['title_embedding']
        body_embedding = record['body_embedding']
        title_body_embedding = record['title_body_embedding']

        title_body_embedding=np.expand_dims(title_body_embedding, axis=0)
        title_embedding = np.expand_dims(title_embedding, axis=0)
        body_embedding=np.expand_dims(body_embedding, axis=0)

        faiss.normalize_L2(title_embedding)
        faiss.normalize_L2(body_embedding)
        faiss.normalize_L2(title_body_embedding)

        title_dist, title_idx= self.title_index.search(title_embedding, topk)
        body_dist, body_idx= self.body_index.search(body_embedding, topk)
        title_body_dist, title_body_idx= self.title_body_index.search(title_body_embedding, topk)

        return (title_dist, body_dist, title_body_dist), (title_idx, body_idx, title_body_idx)

    def post_process(self, records, indices):
        positions = []

        for idx in indices[0]:

            positions.append(records[idx]['idx'])
        return positions

    def get_preds(self, records, gt_records, topk=5):
        res_title, res_body, res_title_body = [], [], []
        for i, record in enumerate(records):
            if i%200==0:
                print('Processed {}/{}'.format(i, len(records)))
            dist, idx = self.search_index(record, topk)
            for title_body in range(len(idx)):
                #dist_idx = dist[title_body]
                idx_idx = idx[title_body]

                result = self.post_process(gt_records, idx_idx)
                if title_body ==0 :
                    res_title.append(result)
                elif title_body ==1:
                    res_body.append(result)
                else:
                    res_title_body.append(result)
        return res_title, res_body, res_title_body

    def predict(self, record, gt_records, topk=5):
        dist, idx = self.search_index(record, topk)

        #using title embedding generates the best model and finds topk relevant ONETS
        res_title_indices =idx[0]
        onets = []
        for idx in res_title_indices[0]:

            onets.append(gt_records[idx]['ONET'])
        return onets







if __name__ == '__main__':
    records = read_data()
    faiss_model = FaissNNeighbors()
    random.seed(42)
    '''
    Uncomment these lines to generate embeddings of the title and/or body of the dataset
    title_embeddings, body_embeddings, title_body_embeddings = faiss_model.get_embeddings(records)
    np.save('embeddings/title_embeddings.npy', title_embeddings)
    np.save('embeddings/body_embeddings.npy', body_embeddings)
    np.save('embeddings/title_body_embeddings.npy', title_body_embeddings)
    '''
    title_embeddings = np.load('embeddings/title_embeddings.npy')
    body_embeddings = np.load('embeddings/body_embeddings.npy')
    title_body_embeddings = np.load('embeddings/title_body_embeddings.npy')

    records, label2idx, label2name = label_to_idx(records)
    records  = append_embeddings(records, title_embeddings, body_embeddings, title_body_embeddings)
    train_records, test_records = split_train_test(records)
    faiss_model.fit(train_records)


    topk=1
    print('getting predictions.....')
    predictions = faiss_model.get_preds(test_records, train_records, topk)
    print('getting gt labels........')
    gt_labels = get_labels(test_records, label2idx)
    get_metrics(predictions, gt_labels)

    topk=7
    onets = faiss_model.predict(record=test_records[0], gt_records=train_records, topk=topk)
    print("\nfor this record, possible ONETs are:", onets)









