from image_loader import *
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import time

#path to the folders
train_path='C:/Users/Hp/Downloads/aerial-cactus-identification/train/'
test_path='C:/Users/Hp/Downloads/aerial-cactus-identification/test/'
train_label_path='C:/Users/Hp/Downloads/aerial-cactus-identification/train.csv'

#labels and id of the datasets
label_df=pd.read_csv(train_label_path)


train_loader=image_loader(train_path)
test_loader=image_loader(test_path)

# load training image without label and can check the time it take to complete loading
start=time.time()
images=train_loader.process_image_no_label()
end_time=time.time()-start

print("time to load training images: ",end_time)

# load test image without label
start1=time.time()
images1=test_loader.process_image_no_label()
end_tim1=time.time()-start1

print("time to load test images: ",end_time)

# this function align each image with its label based on id 
def allign_img_with_label(image,label):
    
    if len(image)<1:
        raise Exception("no items found")
    
    if "id" and "has_cactus" not in label.columns.values:
            raise Exception(f"id or label not found in {label.columns},please rename image name column to id and image label to label ")
    
    index = label.index
    images = []
    labels = []
    for img in image:
        try:
            
            cond=label['id']==img[-1]
            idx=index[cond]
            label_df=label.iloc[idx].values[0,-1]
            images.append(img[0])
            labels.append(label_df)
        
        except IndexError:
            pass
        
    return (np.array(images),np.array(labels))

# this function return test images without its id name
def detach_test_image(image):
        
    if len(image)<1:   
        raise Exception("no items found")
    
    images=[]
    
    for img in image:
        images.append(img[0])
        
    return (np.array(images))


train_x,train_y=allign_img_with_label(images,label_df)
test_x=detach_test_image(images1)

# scaling the images
train_x=train_x/train_x.max()
test_x=test_x/test_x.max()

# lets leave out last 1000 data for validation
new_train_x,new_train_y=train_x[:len(train_x)-1000],train_y[:len(train_y)-1000]
val_x,val_y=train_x[len(train_x)-1000:],train_y[len(train_y)-1000:]

# A DATA PROCESSOR CLASS TO FEED TO THE DATA_LOADER
class train_dataprocess():
    
    def __init__(self,X,Y):
        
        self.len=X.shape[0]
        
        self.x_data=torch.FloatTensor(X.reshape(self.len,3,32,32))
        self.y_data=torch.FloatTensor(Y.reshape(len(Y),1))
        
        
    
    def __getitem__(self,index):
        return(self.x_data[index],self.y_data[index])
    
    def __len__(self):
        return(self.len)


    

DataLoader=torch.utils.data.DataLoader

train_data=train_dataprocess(new_train_x,new_train_y)


#load train and test images
train_loader=DataLoader(dataset=train_data,batch_size=32,shuffle=True)



# this function helps us to keep track of dimensions of each convolution of a signal
def conv_shape(input_shape,channels,kernel=(3,3),max_kernel=(2,2),stride=(1,1),max_stride=(2,2),pad=(0,0),max_pad=(0,0),dilation=(1,1),max_dilation=(1,1),max_pooling=False):
    
    if(len(input_shape) != 2):
        print('expected 2D shape for a convolution')
        return
    
    n_block=len(channels)
    new_output=input_shape
    
    if(max_pooling):
        
        
        for i in range(n_block):
        
            output=pool(new_output,kernel,stride,pad,dilation)
            maxpool=pool(output,max_kernel,max_stride,max_pad,max_dilation)
            new_output=maxpool
            
        
    else:
        
        for i in range(n_block):
            output=pool(new_output,kernel,stride,pad,dilation)
            new_output=output
            
    
    if(new_output[0]<=0 and new_output[1]<=0):
        print('cannot compute for 0 convolutions')
        return
    
    
    return(new_output)
    


def pool(h_w,kernel=(2,2),stride=(2,2),pad=(0,0),dilation=(1,1)):
     
    h=np.floor(((h_w[0] + (2*pad[0]) - (dilation[0]*(kernel[0] - 1) ) - 1)/ stride[0]) + 1)
    w=np.floor(((h_w[1] + (2*pad[1]) - (dilation[1]*(kernel[1] - 1) ) - 1)/ stride[1]) + 1)
    
    return((int(h),int(w)))


#LETS COMPUTE OUTPUT SHAPE USING THE CONV_SHAPE FUNCTION ABOVE
conv_shape1=conv_shape((32,32),channels=[10],kernel=(3,3),max_kernel=(2,2),max_pooling=True)
conv_shape2=conv_shape(conv_shape1,channels=[20],kernel=(3,3),max_kernel=(2,2),max_pooling=True)

#this is the CNN implementation you can modify the architecture for better results
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.cnn1=nn.Conv2d(3,10,kernel_size=(3,3)) # arg: input features,output features
        self.cnn2=nn.Conv2d(10,20,kernel_size=(3,3))
        
        self.mp1=nn.MaxPool2d(2)
        
        self.ffn1=nn.Linear(6*6*20,100)
        
        self.ffn2=nn.Linear(100,1)
        
        self.sigmoid=torch.sigmoid
        
    def forward(self,x):

        output=f.relu(self.mp1(self.cnn1(x)))
        output=f.relu(self.mp1(self.cnn2(output)))

        in_size=x.size(0)
        output=output.view(in_size,-1)
        
        output=torch.tanh(self.ffn1(output))
        
        output=torch.sigmoid(self.ffn2(output))
        

        return(output)

model=CNN()

length=len(new_train_x)
val_len=len(val_x)

#loss function and optimizer
loss_fn=nn.BCELoss()
optim=torch.optim.Adam(model.parameters(),lr=0.01)


#lets train and test our datasets
epoch=30
for i in range(epoch+1):
    correct=0
    for idx,(data_x,data_y) in enumerate(train_loader):
        X,Y=Variable(data_x),Variable(data_y)
        pred=model(X)
        loss=loss_fn(pred,Y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        output=torch.FloatTensor(np.where(pred.data.numpy()<0.5,0,1))
        correct+=(output==Y).float().sum()
        
    if i % 5 == 0:
        
        val_pred=model(torch.FloatTensor(val_x.reshape(-1,3,32,32)))
        val_acc=(torch.FloatTensor(np.where(val_pred.data.numpy()<0.5,0,1))==torch.FloatTensor(val_y.reshape(-1,1))).float().sum()/val_len
        
        acc=100*(correct/length)
        val_acc=100*val_acc

        val_loss=loss_fn(val_pred,torch.FloatTensor(val_y.reshape(-1,1)))
        
        print('\ntraining acc: ',acc)
        print('training loss: ',loss.item())
        print('val acc: ',val_acc)          #the validation accuracy is 97%
        print('val_loss: ',val_loss)
 
 
#lets predict test images
test_images=Variable(torch.FloatTensor(test_x.reshape(-1,3,32,32)))
predictions=np.where(model.predict(test_images).data.numpy()<0.5,0,1)

print("test images predictions: ",predictions)






 
