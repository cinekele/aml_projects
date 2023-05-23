import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_feature_numbers(estimator, X, y):
    est = estimator.named_steps['model']
    return est.n_features_in_


class Objective(object):
    def __init__(self, x, y, feature_selectors, mode: str, scaling_factor: float = None,
                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=123), n_jobs=5, use_scaler=True):
        self.n_jobs = n_jobs
        self.cv = cv
        self.x = x
        self.y = y
        if mode not in ["single", "multiple"]:
            raise Exception("mode isn't single or multiple")
        self.mode = mode
        self.scaling_factor = scaling_factor
        self.use_scaler = use_scaler
        self.feature_selectors = feature_selectors
        self.feature_selector_names = [feature_selector.__name__ for feature_selector in feature_selectors]
        self.feature_selector_names_for_svm = [feature_selector.__name__ for feature_selector in feature_selectors if
                                               feature_selector.__name__ not in {'SelectFromModel', 'RFE'}]

    def __call__(self, trial):
        classifier_name = trial.suggest_categorical('classifier', ['XGB', 'SVC', 'LGBM'])
        if classifier_name == 'XGB':
            booster = trial.suggest_categorical('xgb_booster', ['gbtree', 'dart'])
            max_depth = trial.suggest_int('xgb_max_depth', 1, 15)
            n_estimators = trial.suggest_int('xgb_n_estimators', 10, 500, log=True)
            subsample = trial.suggest_float('xgb_subsample', 0.6, 1)
            classifier_obj = XGBClassifier(booster=booster, max_depth=max_depth,
                                           n_estimators=n_estimators, subsample=subsample,
                                           importance_type='total_gain')
        elif classifier_name == 'SVC':
            kernel = trial.suggest_categorical('svc_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            c = trial.suggest_float('svc_C', 1e-4, 1e4, log=True)
            classifier_obj = SVC(kernel=kernel, C=c)
        else:
            boosting_type = trial.suggest_categorical('lgbm_boosting_type', ['gbdt', 'dart'])
            max_depth = trial.suggest_int('lgbm_max_depth', 1, 15)
            n_estimators = trial.suggest_int('lgbm_n_estimators', 10, 500, log=True)
            subsample = trial.suggest_float('lgbm_subsample', 0.6, 1)
            classifier_obj = LGBMClassifier(boosting_type=boosting_type, max_depth=max_depth,
                                            n_estimators=n_estimators, subsample=subsample,
                                            importance_type='gain')

        if classifier_name != 'SVC':
            fs_name = trial.suggest_categorical('feature_selector', self.feature_selector_names)
        else:
            fs_name = trial.suggest_categorical('feature_selector_svm', self.feature_selector_names_for_svm)

        fs_model = self.feature_selectors[self.feature_selector_names.index(fs_name)]
        if fs_name == "SelectPercentile":
            percentile = trial.suggest_int('percentile', 5, 70)
            fs_model_obj = fs_model(percentile=percentile)
        elif fs_name == "SelectKBest":
            k = trial.suggest_int('k', 5, self.x.shape[1], log=True) # .shape[1] works with sparse matrices
            fs_model_obj = fs_model(k=k)
        elif fs_name == "SelectFromModel":
            fs_model_obj = fs_model(estimator=classifier_obj)
        elif fs_name == "SelectFdr":
            alpha = trial.suggest_float('fdr_alpha', 0.2, 0.5)
            fs_model_obj = fs_model(alpha=alpha)
        elif fs_name == "SelectFpr":
            alpha = trial.suggest_float('fpr_alpha', 0.05, 0.5)
            fs_model_obj = fs_model(alpha=alpha)
        elif fs_name == "SelectFwe":
            alpha = trial.suggest_float('fwe_alpha', 0.05, 0.5)
            fs_model_obj = fs_model(alpha=alpha)
        elif fs_name == "RFE":
            n_features = trial.suggest_float('rfe_n_features', 0.01, 0.6)
            fs_model_obj = fs_model(estimator=classifier_obj, n_features_to_select=n_features)
        else:
            fs_model_obj = fs_model()

        if self.use_scaler:
            pipeline = Pipeline([
                ('st', StandardScaler()), # does not work for sparse matrices
                ('fs', fs_model_obj),
                ('model', classifier_obj)
            ])
        else:
            pipeline = Pipeline([
                ('fs', fs_model_obj),
                ('model', classifier_obj)
            ])

        res = cross_validate(pipeline, self.x, self.y,
                             scoring={
                                 'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                 'feature_numbers': get_feature_numbers
                             }, cv=self.cv, n_jobs=self.n_jobs)

        if self.mode == "multiple":
            return np.mean(res['test_balanced_accuracy']), np.mean(res['test_feature_numbers'])
        else:
            return np.mean(
                res['test_balanced_accuracy'] - 0.01 * np.maximum(
                    self.scaling_factor * res['test_feature_numbers'] - 1, 0))
