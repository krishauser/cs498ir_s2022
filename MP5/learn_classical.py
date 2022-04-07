import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import randint,uniform
import os
import sys
import pickle
from dataset import load_images_dataset,make_patch_dataset

PROBLEM = 'train'
#PROBLEM = 'test'

grasp_attributes = ['score']
#1C: uncomment me to predict more features
#grasp_attributes = ['score', 'axis_heading','axis_elevation','opening']


def train_predictor(X,y):
    #TODO: tune me for problem 1b
    print("Training on dataset with",X.shape[0],"observations with dimension",X.shape[1])
    print("Average score",np.average(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #Select your learning model / pipeline here
    estimators = [('pca', PCA(n_components=20,whiten=True)), ('linear', LinearRegression())]
    # estimators = [('pca', PCA(n_components=20,whiten=True)), ('rf', RandomForestRegressor(random_state=0,n_estimators=10,max_depth=10))]
    # estimators = [('pca', PCA(n_components=20,whiten=True)), ('gb',GradientBoostingRegressor(random_state=0,n_estimators=100,learning_rate=0.1))]
    # estimators = [('pca',PCA(n_components=20,whiten=True)), ('mlp',MLPRegressor(hidden_layer_sizes=(100,)))]

    pipe = Pipeline(estimators)

    pipe.fit(X_train, y_train)
    model = pipe

    # # to do model selection, create a searchCV object and fit it to the data
    # # define the parameter space that will be searched over
    # param_distributions = {'pca__n_components': randint(5,50)}
    # param_distributions = {'pca__n_components': randint(5,50),
    #                         'rf__n_estimators': randint(1, 5),
    #                         'rf__max_depth': randint(5, 10) }
    # param_distributions = {'pca__n_components': randint(5,50),
    #                         'gb__n_estimators': randint(1, 5),
    #                         'gb__learning_rate': uniform(loc=0.3, scale=0.29)}
    # search = RandomizedSearchCV(estimator=pipe,
    #                             n_iter=5,
    #                             param_distributions=param_distributions,
    #                             random_state=0)
    # search.fit(X_train, y_train)
    # print(search.best_params_)
    # model = search
    # the search object now acts like a normal random forest estimator

    print("Test score",model.score(X_test, y_test))
    print("Constant predictor RMSE",np.linalg.norm(y_test-np.average(y_train))/np.sqrt(len(y_test)))
    print("Train RMSE",np.linalg.norm(model.predict(X_train)-y_train)/np.sqrt(len(y_train)))
    print("Test RMSE",np.linalg.norm(model.predict(X_test)-y_test)/np.sqrt(len(y_test)))
    return model

def predict_patches(image_group,pts,model,patch_size=30):
    """Returns predictions of the given model for the given points in
    the image.  Make sure you are calculating features of these
    points exactly like you did in make_patch_dataset.
    """
    #TODO: fill me in for problem 2B
    patch_radius = patch_size//2
    color,depth,transform,grasp_attrs = image_group
    preds = []
    for i,(x,y) in enumerate(pts):
        if len(pts) > 10000 and i%(len(pts)//10)==0:
            print(i//(len(pts)//10)*10,"...")
        roi = (y-patch_radius,y+patch_radius,x-patch_radius,x+patch_radius)
        patch1 = get_region_of_interest(color,roi).flatten()
        patch2 = get_region_of_interest(depth,roi).flatten()
        preds.append(model.predict([np.hstack((patch1,patch2))])[0])
    return preds

def gen_prediction_images(attr='score'):
    """Creates prediction images and saves them to predictions/
    using your predict_patches function.
    """
    import pickle
    with open('trained_model_{}.pkl'.format(attr),'rb') as f:
        model = pickle.load(f)
    dataset = load_images_dataset('image_dataset')
    h,w = dataset[0][1].shape
    patch_radius = 30//2
    X = list(range(patch_radius,w-patch_radius))
    Y = list(range(patch_radius,h-patch_radius))
    pts = np.transpose([np.tile(X, len(Y)), np.repeat(Y, len(X))])
    try:
        os.mkdir('predictions')
    except Exception:
        pass
    img_skip = 10
    for i,image in enumerate(dataset[::img_skip]):
        print("Predicting",len(pts),"for image",i)
        values = predict_patches(image,pts,model)
        pred = np.zeros((h,w))
        for pt,v in zip(pts,values):
            x,y = pt
            pred[y,x] = v
        pred_quantized = (pred*255.0).astype(np.uint8)
        filename = "predictions/image_%04d.png"%(i,)
        Image.fromarray(pred_quantized).save(filename)





if __name__ == '__main__':
    if len(sys.argv) > 1:
        PROBLEM = sys.argv[1]
    if PROBLEM == 'train':
        dataset = load_images_dataset('image_dataset',grasp_attributes)
        
        for attr in grasp_attributes:
            X,y = make_patch_dataset(dataset,attr)
            model = train_predictor(X,y)
            with open('trained_model_{}.pkl'.format(attr),'wb') as f:
                pickle.dump(model,f)
    elif PROBLEM == 'test':
        gen_prediction_images()
    else:
        raise ValueError("Invalid PROBLEM?")

