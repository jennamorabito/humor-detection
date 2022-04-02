# https://towardsdatascience.com/implementing-custom-data-generators-in-keras-de56f013581c

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, x_col, y_col=None, batch_size=1000, num_classes=None, shuffle=True):
        # Define the constructor to initialize the configuration of the generator
        # We assume the path to the data is in a dataframe column,
        # thoughThis could also be a directory name from where you load the data
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = 2
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]

        # Generate data
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        # called after every epoch, with a shuffling routine
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        # X.shape : (batch_size, *dim)
        # We can have multiple Xs and can return them as a list
        X = np.empty((self.batch_size, *self.dim)) # logic to load the data from storage
        y = np.empty((self.batch_size, *self.dim)) # logic for the target variables
        # Generate data
        for i, id in enumerate(batch):
        # Store sample
            X[i,] =  df[x_col][i] # logic
        # Store class
            y[i] = df[y_col][i] # labels
        return X, y
