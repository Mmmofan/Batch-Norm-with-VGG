###VGG_MNIST.py###
import VGG
import tensorflow as tf
import sys
from tensorflow.examples.tutorials.mnist import input_data

vgg = VGG.VGG([None,28,28,1], [100,100,100,10],conv_info=[(2,64),(2,128),(3,256),(3,512),(3,512)], activate_fun=tf.sigmoid)
print("param_num=%d" %(vgg.param_num))

mnist = input_data.read_data_sets("input_data", one_hot=True)

def get_mnist_batch(num, get_test=False):
    batch = None
    if get_test:
        batch = [mnist.test.images, mnist.test.labels]
    else:
        batch = mnist.train.next_batch(num)

    input = []
    for x in batch[0]:
        inp = [[0 for _ in range(0,28)] for _ in range(0,28)]
        for row in range(0,28):
            for col in range(0,28):
                inp[row][col] = [x[row*28 + col]]
                """
                if inp[row][col][0]>0.6:
                    print(" ",end="")
                else:
                if inp[row][col][0]>0.3:
                    print(".",end="")
                else:
                    print("w",end="")
                if col==27:
                    print("")
        sys.exit(0)
        """
        input.append(inp)
    return input, batch[1]

def get_mnist_test_accuracy():
    batch = get_mnist_batch(0, True)
    accuracy = 0
    for st in range(0, 10000, 100):
        ret = vgg.train(batch[0][st:st+100],batch[1][st:st+100], learning_rate=0)
        accuracy += ret["accuracy"]/100
    return accuracy

"""    
if vgg.restore("./model/"):
    test_acc=get_mnist_test_accuracy()
    print("[*]Test Result=%s at epoch%d" %(test_acc,0))
""" 
learning_rate = 1e-4

for epoch in range(0, 10):
    batch_sz = 50
    for i in range(int(50000/batch_sz)):
        batch = get_mnist_batch(batch_sz)
        ret = vgg.train(batch[0], batch[1],learning_rate=learning_rate)
        if i%100==0:
            #print(batch[1][0])
            #print(ret[0][0])
            print("step{0} accuracy={1} loss={2}".format(i+100, ret["accuracy"], ret["loss"]))
    #learning_rate /= 2
    vgg.save("model/mnist_epoch", epoch)
    test_acu = get_mnist_test_accuracy()
    print("[*]Test Result=%s at epoch%d" %(test_acu, epoch))