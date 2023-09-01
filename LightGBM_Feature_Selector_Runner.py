import logging
from DataLoaders.synthetic_dataloader import create_Classification_Dataset
from Models.Feature_Selector_LightGBM import Feature_Selector_LGBM

if __name__ == '__main__':
    logging.basicConfig(filename='console.log', level=logging.DEBUG)
    logging.info('Feature Selector LGBM Log')

    X_train, X_val, X_val_mask, X_test, y_train, y_val, y_val_mask, y_test = create_Classification_Dataset()

    network = Feature_Selector_LGBM(params={"boosting_type": "gbdt","importance_type":"gain",
                                            "verbosity":-1},
                                    param_grid={
                                        'boosting_type': ['gbdt'],
                                        'num_leaves': [20, 50],
                                        'learning_rate': [0.01, 0.1, 0.5],
                                        'n_estimators': [20, 50],
                                        'subsample': [0.6, 0.8, 1.0],
                                        'colsample_bytree': [0.6, 0.8, 1.0],
                                        # 'reg_alpha': [0.0, 0.1, 0.5],
                                        # 'reg_lambda': [0.0, 0.1, 0.5],
                                        'min_child_samples': [5, 10],
                                    },
                                    X_train=X_train, X_val=X_val,
                                    X_val_mask=X_val_mask, X_test=X_test, y_train=y_train,
                                    y_val=y_val, y_val_mask=y_val_mask, y_test=y_test,
                                    data_type="Classification")

    network.fit_network()
