start=datetime.datetime.now()
model = SpectralEmbedding(n_components=num_coord, n_neighbors=neighbors)
#model = SpectralEmbedding(affinity='rbf', gamma=gam, n_components=num_coord, n_neighbors=neighbors)
laply_new=model.fit_transform(data)
end=datetime.datetime.now()
print("Laplacian Eigenmap Time Taken: " + str(end-start))

if doPlot:
	fig = plt.figure(figsize=(8,6))
	ax=Axes3D(fig)
	plt.figtext(.5,.95,title,fontsize=25,ha='center')
	ax.scatter(laply_new[:, 0], laply_new[:, 1], laply_new[:, 2], c = classes, cmap=plt.cm.Spectral)
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

# kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(mnist_diff)
# for i in range(num_classes):
#     total = 0
#     for x in kmeans.labels_:
#         if x == i: total += 1
#     print(total)

X_train, X_test, y_train, y_test=train_test(laply_new, classes)

if doKNN:
	# training a KNN classifier
	start=datetime.datetime.now()
	knn = KNeighborsClassifier(n_neighbors = 20).fit(X_train, y_train)
	  
	# accuracy on X_test 
	accuracy = knn.score(X_test, y_test) 
	print("KNN Accuracy: " + str(accuracy)) 

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