# RAG_LangChain

## Contents

1. ChromaDB
2. LangChain & OpenAI

A simple workflow of creating a DB out of a .pdf document and querying over that context.

The script first returns a direct response from chatGPT and then a refined one, after looking at the context DB.

## Example query

### Query: Give me an algorithm for local feature matching between images. Explain the steps in detail

### Answer

Direct response: Algorithm for local feature matching between images:

1. Extract local features from both images using a feature detection algorithm such as SIFT (Scale-Invariant Feature Transform) or ORB (Oriented FAST and Rotated BRIEF).

2. Compute descriptors for each keypoint in both images. Descriptors are numerical representations of the local features that capture important information about the keypoints.

3. Match the descriptors between the two images using a matching algorithm such as nearest neighbor matching or brute force matching. This involves comparing the descriptors of keypoints in one image with the descriptors of keypoints in the other image and finding the best match for each keypoint.

4. Apply a distance metric, such as Euclidean distance or Hamming distance, to measure the similarity between descriptors and find the best match for each keypoint.

5. Use a matching criterion, such as ratio test or thresholding, to filter out incorrect matches and retain only the most reliable matches.

6. Optionally, perform geometric verification to further refine the matches by considering the spatial relationship between keypoints in the two images.

7. Output the matched keypoints and their corresponding keypoints in the two images.

Explanation of steps:

1. Local feature extraction: This step involves detecting keypoints or interest points in the images and extracting local features around these keypoints. These features are invariant to changes in scale, rotation, and illumination, making them suitable for matching between images.

2. Descriptor computation: Descriptors are numerical representations of the local features that capture important information about the keypoints, such as their shape, texture, and intensity. These descriptors are used to compare and match keypoints between images.

3. Matching: The matching step involves comparing the descriptors of keypoints in one image with the descriptors of keypoints in the other image and finding the best match for each keypoint. This step is crucial for finding correspondences between keypoints in different images.

4. Distance metric: A distance metric is used to measure the similarity between descriptors and find the best match for each keypoint. Common distance metrics include Euclidean distance and Hamming distance, depending on the type of descriptors used.

5. Matching criterion: A matching criterion, such as a ratio test or thresholding, is applied to filter out incorrect matches and retain only the most reliable matches. This helps to reduce the number of false matches and improve the accuracy of the matching process.

6. Geometric verification: Geometric verification can be optionally performed to further refine the matches by considering the spatial relationship between keypoints in the two images. This step helps to eliminate outliers and improve the overall quality of the matches.

7. Output: The final output of the algorithm is a set of matched keypoints and their corresponding keypoints in the two images. These correspondences can be used for various computer vision tasks such as image stitching, object recognition, and 3D reconstruction.


Looking at the database...


Response: The algorithm for local feature matching between images involves the following steps:

1. Split the images into several horizontal parts to extract local features.
2. Create a matrix where each element corresponds to the distance between a part of Image A and a part of Image B.
3. Normalize the distances using an element-wise transformation.
4. Calculate the distance between the two images based on the local parts using Euclidean distance.
5. Use the following equation to calculate the distance between local features:
   di,j = e||fi - gj||^2 - 1 / (e||fi - gj||^2 + 1)
   where fi is the local feature i of image A, and gj is the local feature j of image B.
6. Find the shortest path in the distance matrix to determine the overall distance between the two images, considering alignments between corresponding parts.
7. Use the Triplet Loss function, which compares the embeddings of three images (anchor, positive, negative) to improve the feature representation.
8. Evaluate the performance of the algorithm using metrics such as AUC, rank-1, rank-5, and mAP.

These steps outline the process of matching local features between images using the algorithm described in the provided context.

Sources: {'data/vaska579.pdf'}
