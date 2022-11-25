import numpy as np
from struct import unpack
import os
import gzip

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4),dtype=dt)[0]

def extract_images(filename):
    """Extract the image into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting',filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051 :
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype = np.uint8)
        data = data.reshape(num_images, 1, rows, cols)
        return data

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(filename, one_hot=False):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

def read_data_sets(train_dir, one_hot=False, dtype = np.float32):
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS ='t10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    local_file = os.path.join(train_dir, TRAIN_IMAGES)
    train_images = extract_images(local_file)
    local_file = os.path.join(train_dir, TRAIN_LABELS)
    train_labels = extract_labels(local_file, one_hot = one_hot)
    local_file = os.path.join(train_dir, TEST_IMAGES)
    test_images = extract_images(local_file)
    local_file = os.path.join(train_dir, TEST_LABELS)
    test_labels = extract_labels(local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    train = [train_images / 255.0, train_labels]
    valid = [validation_images / 255.0, validation_labels]
    test = [test_images / 255.0, test_labels]
    return train, valid, test

class Conv():
    def __init__(self, input_channels, filters, kernel = 5, feature_mapping = None, activation = 'relu'):
        self.input_channels = input_channels
        self.filters = filters
        self.kernel = kernel
        self.feature_mapping = feature_mapping
        if self.feature_mapping is None:
            self.feature_mapping = np.ones((self.filters, self.input_channels))
        self.activation = activation
        self.init_params()

    def init_params(self):
        self.w = np.random.randn(self.filters, self.input_channels, self.kernel, self.kernel) / np.sqrt(self.input_channels)
        for i in range(self.filters):
            for j in range(self.input_channels):
                if not self.feature_mapping[i][j]: self.w[i][j] = 0
        self.b = np.zeros(self.filters)

    def conv(self, x, w, mapping=None):
        dim = np.subtract(x[0].shape, w[0].shape)+1
        if mapping is None: mapping = np.ones(x.shape[0])
        a=np.zeros(dim)
        for i in range(dim[0]):
            for j in range(dim[1]):
                p = np.multiply(x[:,i:i+w.shape[1],j:j+w.shape[2]],w).sum((1,2))
                a[i][j] = np.sum(p * mapping)
        return a

    def forward(self, x):
        self.x = x
        a = []
        for i in range(self.filters):
            self.z = self.conv(x, self.w[i], self.feature_mapping[i])+self.b[i]
            if self.activation == 'relu':
                a.append(np.maximum(0,self.z))
            else:
                a.append(self.z)
        self.a = np.array(a)
        return self.a

    def backward(self, delta, eta):
        if self.activation == 'relu':
            delta = delta * (self.z >= 0)
        d = np.pad(delta, ((0,), (self.kernel - 1,), (self.kernel - 1,)), mode='constant', constant_values=0)
        ds=[]
        for i in range(self.input_channels):
            w = np.array([np.rot90(np.rot90(self.w[j][i])) for j in range(self.filters)])
            ds.append(self.conv(d,w))
        ds = np.array(ds)

        for i in range(self.filters):
            d = np.array([delta[i]])
            self.b[i] -= eta * np.sum(d)
            for j in range(self.input_channels):
                if not self.feature_mapping[i][j]: continue
                x = np.array([self.x[j]])
                dw = self.conv(x, d, self.feature_mapping[i][j])
                self.w[i][j] -= eta * dw
        return ds

class Pool():
    def __init__(self, channels, size=2, stride=2):
        self.channels = channels
        self.size = size
        self.stride = stride

    def forward(self, x):
        iw, ih = x.shape[1:]
        ow, oh = iw // self.stride, ih // self.stride
        z = np.zeros((self.channels, ow, oh))
        self.pos = np.zeros((self.channels,ow,oh),dtype = np.int)
        for i in range(self.channels):
            for j in range(0, iw, self.stride):
                for k in range(0, ih, self.stride):
                    z[i,j // self.stride,k // self.stride]=np.max(x[i,j:j + self.size,k:k + self.size])
                    self.pos[i,j // self.stride,k // self.stride] =np.argmax(x[i, j:j + self.size, k:k + self.size])
        return z

    def backward(self, delta):
        iw, ih = delta.shape[1:]
        ow, oh = iw * self.stride, ih * self.stride
        d = np.zeros((self.channels, ow, oh))
        for i in range(self.channels):
            for j in range(0, iw):
                for k in range(0, ih):
                    d[i,j * self.stride + self.pos[i,j,k] // self.size,k * self.stride + self.pos[i,j,k] % self.size] = delta[i, i, k]
        return d
class FullyConnect():
    def __init__ (self, input_size, output_size, activation='relu'):
        self.input_size =input_size
        self.flat_input_size =np.product(input_size)
        self.output_size =output_size
        self.activation =activation
        self.init_params()
    def init_params(self):
        self.w=np.random.randn(self.flat_input_size, self.output_size)/np.sqrt(self.flat_input_size)
        self.b=np.zeros(self.output_size)

    def forward(self, x):
        self.x = x
        #.reshape(self.flat_input_size)
        z = np.dot(self.x, self.w)+self.b
        if self.activation=='relu':
            a=np.maximum(0,z)
        else:
            a = z
        return a
    def backward(self,delta,eta):
        dw=np.outer(self.x,delta)
        db = delta
        d=np.dot(delta,self.w.transpose())
        self.w -= eta *dw
        self.b -= eta *db
        return d.reshape(self.input_size)

class softmax():
    def __init__ (self, size):
        self.size =size
    def forward(self, x):
        self.x=x.reshape(self.size)
        e=np.exp(self.x-np.max(self.x))
        self.a=e/np.sum(e)
        return self.a
    def backward(self,delta):
        i=np.argmax(self.a)
        m=np.zeros((self.size,self.size))
        m[:,i]=1
        m=np.eye(self.size)-m
        d=np.diag(self.a)-np.outer(self.a, self.a)
        d=np.dot(delta,d)
        d =np.dot(d, m)
        return d

class MSE():
    def loss(y_true, y_pred):
        return np.sum(np.square(y_true-y_pred))/2
    def derivative(y_true, y_pred):
        return y_pred-y_true

class LogLikelihood():
    def loss(y_true, y_pred):
        loss =np.sum(y_true * y_pred)
        loss =-np.log(loss) if loss !=0 else 500
        return loss

    def derivative(y_true, y_pred):
        d = y_pred.copy()
        d[np.argmax(y_true)] -= 1
        return d
class LeNet5():
    def __init__ (self,input_size):
        self.input_size=input_size
        self.c1=Conv(input_size[0],6)
        self.s2 =Pool(6)
        self.c3=Conv(6,16)
        self.s4 =Pool(16)
        self.c5=FullyConnect((16,44),120)
        self.f6 =FullyConnect(120,84)
        self.output = FullyConnect(84,10)
        self.softmax = softmax(10)

    def forward(self,x):
        out = self.c1.forward(x)
        out = self.s2.forward(out)
        out = self.c3.forward(out)
        out = self.s4.forward(out)
        out = self.c5.forward(out)
        out = self.f6.forward(out)
        out = self.output.forward(out)
        out = self.softmax.forward(out)
        return out

epochs = 1
shuffle = True
lr = 0.001

def main():
    train, valid, test = read_data_sets('./', one_hot=True)
    train_input = train[0]
    train_label = train[1]
    val_input = valid[0]
    val_label = valid[1]
    test_input = test[0]
    test_label = test[1]

    seq = np.arange(len(train_input))
    net = LeNet5(train_input[0].shape)

    for epoch in range(epochs):
        if shuffle: np.random.shuffle(seq)
        for step in range(100):
            i = seq[step]
            x = train_input[i]
            y_true = train_label[i]
            y = net.forward(x)
            loss = LogLikelihood.loss(y_true, y)
            dloss = LogLikelihood.derivative(y_true, y)
            d = net.backward(dloss, lr)

            if step > 0 and step % 99 == 0:
                print('Epoch %d step %d loss %f' % (epoch, step, loss))
                correct = 0
                loss = 0
                for i in range(len(test_input)):
                    x = test_input[i]
                    y_true = test_label[i]
                    y = net.forward(x)
                    loss += LogLikelihood.loss(y_true, y)
                    if np.argmax(y) == np.argmax(y_true): correct +=1
                print('Test accuracy: %.2f%%, average loss: %f' % (correct/len(test_input)*100, loss/len(test_input)))

if __name__ == '__main__':
    main()