from UniSVM import UniSVM
import time
import yaml

if __name__ == '__main__':

    # define all the loss functions to train, for more loss functions, see LossFunction.py
    # define every loss function's args, can be modified in the LossConfig.yml file
    loss_config = open('Config/LossConfig.yml')
    loss_dict = yaml.load(loss_config)
    # define the training args in TrainConfig.yml
    train_config = open('Config/TrainConfig.yml')
    train_args = yaml.load(train_config)
    # generate the model
    u_svm = UniSVM()
    # load in the train data
    u_svm.dataload('Data/a9a', mode='train')
    # load in the test data
    u_svm.dataload('Data/a9a_test', mode='test')
    print('Training set: %d, Test set: %d' % (u_svm.data.shape[0], u_svm.test_data.shape[0]))

    # To train the model with all the loss functions
    # To test all the model
    for t in loss_dict:
        start = time.time()
        u_svm.train(lambda_=train_args['lambda_'], limit=train_args['limit'],
                    ker_para=train_args['ker_para'], subsetSize=train_args['subsetSize'],
                    errorBound=train_args['errorBound'], str=list(t.keys())[0], args=list(t.values())[0])
        times = time.time() - start
        u_svm.test()
        print('UniSVM Test Accuracy:%.2f %% time: %.2f , Iter: %d, lossName: %s' %
              (u_svm.testAccuracy, times, u_svm.iteration, t.keys()))
