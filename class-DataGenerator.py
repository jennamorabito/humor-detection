# https://towardsdatascience.com/implementing-custom-data-generators-in-keras-de56f013581c
from transformers import BertTokenizer
from bertembeddings import compute_input_arrays

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col, y_col=None, batch_size=1000, num_classes=1, shuffle=True):
        # Define the constructor to initialize the configuration of the generator
        # We assume the path to the data is in a dataframe column,
        # thoughThis could also be a directory name from where you load the data
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_col = x_col # name of column containing data (list of str)
        self.y_col = y_col # name of column containing labels (str)
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size] # batched shuffled indices
        # Find list of IDs
        batch = [self.indices[k] for k in index]

        # Generate data
        X, y = self.__get_data(batch)
        # save processed data to npz file
        X_dict = dict(zip(map(str, range(len(X))), X))
        X_dict['y'] = y
        np.savez_compressed("batch_{}_inputs_{}.npz".format(batch[0], self.batch_size), **X_dict)
        
        return X, y

    def on_epoch_end(self):
        # called after every epoch, with a shuffling routine
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        # X.shape : (batch_size, *dim)
        # We can have multiple Xs and can return them as a list
        # Generate data
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        X = compute_input_arrays(self.df.loc[batch, x_col], x_col, tokenizer)
        y = self.df.loc[batch, 'humor']
        return X, y
