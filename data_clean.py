
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

def transform_dataset(data_set_df, target_col="Churn", test_size=0.3, random_state=42,clf=False,drop=None):
    df=data_set_df.copy()
    y=data_set_df["Churn"]
    X=data_set_df.drop(columns=["Churn"])
    if drop != None:
        X = X.drop(columns=drop)
    X=X.drop(columns=["Age Group"])  # always drop Age Group - assume age is enough
    
    #  Define columns by type 

    continuous_and_counts = [
        "Subscription Length",
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Age",
        "Customer Value",
        "Call Failure",
    ]
   

    binary_categoricals_2level = [
        "Complaints",   # 0/1 already
        "Tariff Plan",  # 1=Pay as you go, 2=contractual -> map to 0/1
        "Status",       # 1=active, 2=non-active     -> map to 0/1
    ]

    ordinal_categoricals = [
        "Charge Amount",  # 0..9
    ]

     

    #  basic Cleaning and Mapping

    # Change mapping to 0/1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!I donth think thus us working !!!!!!!!!!!!!!!!!!!!!!!
    df["Tariff Plan"] = df["Tariff Plan"].map({1: 0, 2: 1})  # PAYG=0, Contractual=1
    df["Status"]      = df["Status"].map({1: 0, 2: 1})       # Active=0, Non-active=1
    if drop != None:
        continuous_and_counts = [col for col in continuous_and_counts if col not in drop]
        binary_categoricals_2level = [col for col in binary_categoricals_2level if col not in drop]
        ordinal_categoricals = [col for col in ordinal_categoricals if col not in drop]
    # define train and tests memory

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    #continue only if needed
    if clf==False:
        print("Returning with no clf")
        return X,y,X_train, X_test, y_train, y_test

    # 4) Preprocessing by type
    #    - numeric & counts: impute median, then scale
    #    - binary 0/1: impute most_frequent, no scaling
    #    - ordinal: impute most_frequent, then (optionally) scale
    #      - if we want to treat it as continuous, we could use a scaler

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")), 
        ("scale", StandardScaler()),
    ])

    bin_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    ord_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])



    

    

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, continuous_and_counts),
            ("bin", bin_pipe, binary_categoricals_2level),
            ("ord", ord_pipe, ordinal_categoricals),
        ],
        remainder="drop",
    )

    #feature names for logistic regression
    preprocess.fit(X_train)  # fit only preprocessing (not the model yet)
    X_train_transformed = preprocess.transform(X_train)
    feature_names = preprocess.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    #5 Set up Model
    log_reg = LogisticRegression(
        class_weight="balanced",   # helpful if churners are rarer 
        penalty="l2", 
        C=1.0, 
        solver="lbfgs", 
        max_iter=1000
    )

    clf = Pipeline(steps=[
        ("prep", preprocess),
        ("model", log_reg),
    ])

    return (X,y,X_train, X_test, y_train, y_test, clf,ordinal_categoricals,binary_categoricals_2level,continuous_and_counts, X_train_df)