import torch
import torchvision
import torch.nn as nn
import numpy as np
import os
from core.model import simple_classifier, load_pretrained_classifier
from core.data_loader import get_train_loader, get_test_loader
from tqdm import tqdm
import torch.optim as optim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_classifier(testloader, classifier, max_iter=2000):
    correct = 0
    total = 0
    classifier.eval()
    with torch.no_grad():
        for ii, data in enumerate(tqdm(testloader)):
            if ii >= max_iter:
                break
            images, labels = data[0].to(device), data[1].to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (
        total, 100 * correct / total))
    return correct / total


if __name__ == '__main__':
    # Data parameters:
    data_name = 'afhq' # data's folder name
    img_channels = 3 # 3 or 1
    img_size = 256
    batch_size = 8

    # Training parameters:
    lock_test_mode = False 
    training_mode = False
    use_ckpt = False
    epochs = 100
    lr = 1e-4

    # Classifier parameters:
    classifier_type = 'resnet18'
    classifier_path = './' + data_name + '_' + classifier_type + '_ch_' + str(img_channels) + '_weights.ckpt'

    train_img_dir = '../Data/'+data_name+'/train'
    val_img_dir = '../Data/'+data_name+'/val'
    len_train_set = sum([len(files) for r, d, files in os.walk(train_img_dir)])
    len_test_set = sum([len(files) for r, d, files in os.walk(val_img_dir)])

    # Define the data-loaders:
    trainloader = get_train_loader(root=train_img_dir,
                                     img_size=img_size,
                                     batch_size=batch_size,
                                     num_workers=2,
                                     img_channels=img_channels)
                                     
    testloader = get_test_loader(root=val_img_dir,
                                 img_size=img_size,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 img_channels=img_channels)

    classes = sorted(os.listdir(val_img_dir))
    num_of_classes = len(classes)

    # Build the classifier:
    if classifier_type in 'simple_classifier':
        classifier = simple_classifier(img_size=img_size, img_channels=img_channels, num_domains=num_of_classes)
    elif classifier_type in 'resnet18':
        classifier = torchvision.models.resnet18(pretrained=False)
        nr_filters = classifier.fc.in_features  # number of input features of last layer
        classifier.fc = nn.Linear(nr_filters, num_of_classes)
        classifier.conv1 = nn.Conv2d(img_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif classifier_type in 'resnet50':
        classifier = torchvision.models.resnet50(pretrained=False)
        nr_filters = classifier.fc.in_features  # number of input features of last layer
        classifier.fc = nn.Linear(nr_filters, num_of_classes)
        classifier.conv1 = nn.Conv2d(img_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif classifier_type in 'vgg11':
        classifier = torchvision.models.vgg11(pretrained=False)
        classifier.classifier[6] = nn.Linear(in_features=4096, out_features=num_of_classes)
        classifier.features[0] = nn.Conv2d(img_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    classifier.to(device)
        
    print('############################################')
    print('data name:                   ', data_name)
    print('classes:                     ', classes)
    print('number of classes:           ', num_of_classes)
    print('number of train examples:    ', len_train_set)
    print('number of test examples:     ', len_test_set)
    print('number of channels:          ', img_channels)
    print('number of epochs:            ', epochs)
    print('images size:                    ', img_size)
    print('classifier path is:                     ', classifier_path)

    # Load the classifier if its already exist:
    if os.path.isfile(classifier_path):
        classifier.load_state_dict(torch.load(classifier_path))

    acc = test_classifier(testloader, classifier, len_test_set)

    # If the accuracy is too low, activate training mode:    
    if not lock_test_mode:
        training_mode = True if acc < 0.9 else False
            
    if training_mode:
        
        # Load checkpoint if trainig from scratch is not neccesary: 
        if use_ckpt and os.path.isfile(classifier_path): 
            classifier.load_state_dict(torch.load(classifier_path))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        
        # Defining variables to stop training and prevent overfitting:
        max_val_acc = max(0, acc)
        count_to_stop = 0
        
        print('Training...')
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0
            
            # Check the accuracy on 10% of the validation, if the accuracy improves keep the weights. 
            # If there is no improvement for 3 epochs or the accuracy decreased by more than 5%, end the training.
            curr_acc = test_classifier(testloader, classifier, max_iter=np.round(len_test_set/10/batch_size))
            if max_val_acc < curr_acc and curr_acc<0.999:
                max_val_acc = curr_acc;
                torch.save(classifier.state_dict(), classifier_path)
                print('chkpt saved')
                print('max validation accuracy updated to: %.3f'  %(max_val_acc))
            elif max_val_acc - curr_acc > 5 or count_to_stop>3 or curr_acc>=0.999:
                break
            elif max_val_acc - curr_acc <=5 and curr_acc<0.999:
                count_to_stop += 1
                     
        print('Finished Training')

        # Check the total accuracy:
        classifier.load_state_dict(torch.load(classifier_path))
        acc = test_classifier(testloader, classifier, len_test_set)

    print('Done')
