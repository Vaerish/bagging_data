setStuff = pd.read_csv("C:\\Users\\shado_000\\hw8_data.csv")

Y = setStuff.iloc[:,0].values
X = setStuff.iloc[:,1:38]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

model = BaggingClassifier(DecisionTreeClassifier(max_depth = None), n_estimators=500, random_state=0)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
