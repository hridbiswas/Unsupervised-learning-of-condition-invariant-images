clear all;
close all;
clc;
im_dimension=[26,64];
matrix=[];
srcFiles = dir('F:\DATASETS\Alderly_Day_Night_DAtaset\for training_combined_2500\*.jpg');  
for j = 1 : length(srcFiles)
filename = strcat('F:\DATASETS\Alderly_Day_Night_DAtaset\for training_combined_2500\',srcFiles(j).name);
    I = imread(filename);
  
      gray=rgb2gray(I);
      gray_resize=imresize(gray,(im_dimension));
      
      feature_vector=gray_resize(:);
      
      matrix_cat=horzcat(matrix,feature_vector);
      matrix=matrix_cat;
 
end
    matrix_winter=matrix;
    
t=matrix_winter';
mu=mean(t);
Normalized_feature_matrix=double(t)-double(mu);
A=Normalized_feature_matrix;
cov=A'*A;


[v,d]=eig(cov); % v here vector and d represents eig value
dval=diag(d);

[dvalsort, index]=sort(dval,'descend');
 sortedeigenvalue=dvalsort;
 
 
 vsort=v(:,index); %sorted eigenvector 2048*2048
vsort2=v(:,index(1:300));
scores=A*vsort2;

%%plotting eigenfaces
figure;
for n = 1:6
    subplot(2,3,n);
    evector = reshape(vsort2(:,n),im_dimension);
    
    img_adjust=imadjust(evector); % intensity scaling
    imshow(img_adjust);
   
end  
suptitle('EigenFaces from PCA')
nComp = 100;
Xhat = scores(:,1:nComp) * vsort2(:,1:nComp)'; %2500*100 weight vector for each image
Xhat_new = bsxfun(@plus, Xhat, mu);
m=Xhat_new(1,:);
n=Xhat_new(2,:);
o=Xhat_new(3,:);

%%check training performance 
figure;
subplot(131);
recon1 = reshape(uint8(m),im_dimension);
imshow(imadjust(recon1)),title('PCA Recon_1');
subplot(132)
recon2 = reshape(uint8(n),im_dimension);
imshow(recon2),title('PCA Recon_1');
subplot(133)
recon3 = reshape(uint8(o),im_dimension);
imshow(recon3),title('PCA Recon_1');
suptitle('Reconstruction of training images by PCA')



%Day_test image -PCA
 %for 1:100 eigenVectors
p=imread('Day_12180.jpg');
 p_gray=rgb2gray(p);
 p_resize=imresize(p_gray,im_dimension);
 figure;
 subplot(3,2,1)
 imshow(p_resize);
 title('Day')
 q=imread('night_14366.jpg');
 q_gray=rgb2gray(q);
 q_resize=imresize(q_gray,im_dimension);
 subplot(3,2,2)
 imshow(q_resize);
 title('Night')
 p1=p_resize(:);
 scores_p1_day=double(p1)'*vsort2;
 p1_day_projection=(scores_p1_day(1:100)*vsort2(:,1:100)')+mu;
 recon_day = reshape(uint8(p1_day_projection),im_dimension);
 subplot(3,2,3)
 
 imshow(recon_day);
 title('variant')
 %Night_test_image-PCA
  %for 1:100 eigenVectors
 
 q1=q_resize(:);
 scores_q1_night=double(q1)'*vsort2;
 q1_night_projection=(scores_q1_night(1:100)*vsort2(:,1:100)')+mu;
 recon_night = reshape(uint8(q1_night_projection),im_dimension);
 subplot(3,2,4)
 
 imshow(recon_night);
 title('variant')
 %Day_test image 
 %for 101:200 eigenVectors
 p1_day_projection=(scores_p1_day(101:200)*vsort2(:,101:200)')+mu;
 recon_day_invariant = reshape(uint8(p1_day_projection),im_dimension);
 subplot(3,2,5)
 
 imshow(recon_day_invariant);
 title('invariant')
 %night_test_image
  %for 101:200 eigenVectors
  q1_night_projection=(scores_q1_night(101:200)*vsort2(:,101:200)')+mu;
 recon_night_invariant = reshape(uint8(q1_night_projection),im_dimension);
 subplot(3,2,6)

 imshow(recon_night_invariant);
 title('invariant')
suptitle('Reconstruction of TEST images using PCA')

