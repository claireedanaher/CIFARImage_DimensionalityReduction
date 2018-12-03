start=datetime.datetime.now()
neighbor_params = {'n_jobs': -1, 'algorithm': 'kd_tree'}
mydmap = dm.DiffusionMap(n_evecs=num_coord, k=neighbors, epsilon=eps, alpha=alph, neighbor_params=neighbor_params)
X_new = mydmap.fit_transform(data)
end=datetime.datetime.now()
print("Diffusion Map Time Taken: " + str(end-start))

if doPlot:

	# fig = plt.figure()
	# plt.title("2D Projection")
	# plt.scatter(X_new[:, 0], X_new[:, 1], c = classes, cmap=plt.cm.Spectral)
	# plt.axis('tight')
	# plt.xticks([]), plt.yticks([])
	# plt.show()


	fig = plt.figure(figsize=(8,6))
	ax=Axes3D(fig)
	plt.figtext(.5,.95,title,fontsize=25,ha='center')
	ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c = classes, cmap=plt.cm.Spectral)
	plt.axis('tight')
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	plt.show()
	
def train_test(data,labels):    
	from sklearn.model_selection import train_test_split 
	df_2 = pd.DataFrame(data)
	X = df_2
	y = labels
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 
	
	return(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test=train_test(X_new, classes)

if doKNN:
	# training a KNN classifier
	start=datetime.datetime.now()
	knn = KNeighborsClassifier(n_neighbors = 20).fit(X_train, y_train)
	  
	# accuracy on X_test 
	accuracy = knn.score(X_test, y_test) 
	print("KNN ACCURACY: " + str(accuracy)) 

	# creating a confusion matrix 
	# knn_predictions = knn.predict(X_test)  
	# cm = confusion_matrix(y_test, knn_predictions)
	# print(cm)
	
	end=datetime.datetime.now()
	print("KNN Time Taken: " + str(end-start))

if doKNN or doSVM:
	print('\n##########################\n')
else:
	print('\n')

if doSVM:
	start=datetime.datetime.now()
	svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
	svm_predictions = svm_model_linear.predict(X_test) 
	  
	# model accuracy for X_test   
	accuracy = svm_model_linear.score(X_test, y_test) 
	print("SVM Accuracy: " + str(accuracy)) 
	  
	# creating a confusion matrix 
	cm = confusion_matrix(y_test, svm_predictions) 
	print(cm)
	
	end=datetime.datetime.now()
	print("SVM Time Taken: " + str(end-start))