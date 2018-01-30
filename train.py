import chainer
from chainer import function as F
from chainer import Variable,cuda,optimizers,serializers
from make_gaussian import make_dataset
from cpm import CPM
import numpy as np
from PIL import Image
import os

GPU = 0

net = CPM(n_point=5,n_stage=3)

xp = np
if GPU>=0:
    cuda.get_device(GPU).use()
    xp = cuda.cupy
    net.to_gpu(GPU)

optimizer = optimizers.Adam(alpha=1e-4)
optimizer.setup(net)
x,t = make_dataset('dataset/')


for epoch in range(10000):
    print(epoch)
    if epoch % 10 == 0:
        os.mkdir('Results/'+str(epoch))

    for i in range(len(x)):
        x_train = Variable(xp.asarray([x[i]],dtype=np.float32))
        t_train = Variable(xp.asarray([t[i]],dtype=np.float32))
        y,loss = net(x_train,t_train)
  
        if epoch % 10 == 0: 
          x_image = Image.fromarray(np.uint8(cuda.to_cpu(x_train.data)[0].transpose(1,2,0)*255.))
          x_image.save('Results/'+str(epoch)+'/'+str(i)+'.png') 

 
        y_cpu = cuda.to_cpu(y.data)[0] 
        y_cpu = np.clip(y_cpu,0.0,1.0)

        if epoch % 10 == 0:
          y_image = np.zeros((1,y.data.shape[2],y.data.shape[3]))

          for j in range(y_cpu.shape[0]):
            y_image += y_cpu[j]

          y_image = np.clip(y_image,0.0,1.0)
          y_image = np.tile(y_image,(3,1,1))
          y_image = Image.fromarray(np.uint8(y_image.transpose(1,2,0)*255.))
          y_image = y_image.resize((y.data.shape[3]*8,y.data.shape[2]*8))
          y_image.save('Results/'+str(epoch)+'/'+str(i)+'_a.png')

        print('t    data')
        print('     max :',y_cpu.max())
        print('     min :',y_cpu.min())
        print('loss data:',loss.data)

        net.cleargrads()
        loss.backward()
        optimizer.update()

    print('Saving model...')
    serializers.save_npz("models/mymodel-"+str(epoch)+".npz", net)
