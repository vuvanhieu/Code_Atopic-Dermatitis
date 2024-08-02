## Model Development

Let \(X\) represent the input data set containing images represented as feature vectors and \(Y\) represent the corresponding labels of the images, where \( Y_i \) is the image label \( i \) and \( Y_i \in \left\{ 1, 2 \right\} \) corresponds to "AD", "CD".

Build a predictive model using K-means clustering technique \cite{Hartigan1979AKC} combined with Deep Neural Networks (CNN) to achieve accurate disease image classification into either types: "AD", "CD".

The proposed model, called Cluster-CNN, is illustrated in three phases as depicted in Figure \ref{fig:proposed}.

- **Phase 1:**
  - **Step 1:** Collect disease images and organize them into respective disease image folders.
  - **Step 2:** Extract deep learning features from the images using the VGG-16 model.
  - **Step 3:** Store the resulting feature vectors in the feature database.
- **Phase 2:**
  - **Step 4:** Cluster the feature vectors and determine the number of matching clusters.
  - **Step 5:** Train each cluster using a deep CNN model.
- **Phase 3:**
  - **Step 6:** Combine the trained models from each cluster using a combined classifier.

<img src="images/fig2_ad_proposed.png" alt="Sample Image" width="600" height="400">


## Training and Evaluation

### Training Process and Hyperparameters

Given \(\left\{ X_{\text{cluster}}, Y_{\text{cluster}} \right\}_{1}^M\) is the matrix of samples and the corresponding labels in each cluster \(c\), where \(c= \overline{1,M}\), \(M\) is the number of clusters. Here, \(N\) refers to the number of samples within the validation set, and \(T\) is the number of samples within the test set. Thus, \(Validation~Set\) and \(Test~Set\) are comprised of the feature matrix and labels associated with the samples within the validation set and the corresponding test set. These are defined as in Equation (\ref{eq:1}) and Equation (\ref{eq:2}).

\[
\begin{equation}
\label{eq:1}
Validation~Set = \left\{ X_{\text{val}}, Y_{\text{val}} \right\}_{1}^{N}
\end{equation}
\]

\[
\begin{equation}
\label{eq:2}
Test~Set = \left\{ X_{\text{test}}, Y_{\text{test}} \right\}
\end{equation}
\]

\(Stack~Set\) denotes the amalgamated database containing predictions and their corresponding labels for the collective set of \(M\) predictions, as expressed in Equation (\ref{eq:Stack}).

\[
\begin{equation}
\label{eq:Stack}
Stack~Set = \left\{ X_{\text{stacked}}, Y_{\text{stacked}} \right\}_{1}^{T}
\end{equation}
\]

Given \(W\) are the weights for the association model, the primary objective of the problem is to determine the optimal values of \(W\) that minimize the loss function of the association model when applied to the validation set, as depicted in Equation (\ref{eq:3}). Simultaneously, the aim is to attain the highest achievable accuracy of the combined model when evaluated on the test set, as presented in Equation (\ref{eq:4}).

\[
\begin{equation}
\label{eq:3}
\underset{\mathbf{W}}{\min}~\mathbf{Loss}\left( \mathbf{W} , Validation~Set \right)
\end{equation}
\]

Furthermore, the objective to achieve the highest accuracy of the association model on the test set can be expressed as follows:

\[
\begin{equation}
\label{eq:4}
\underset{\mathbf{W}}{\max}~\mathbf{Accuracy}\left( \mathbf{W} , Test~Set \right)
\end{equation}
\]
