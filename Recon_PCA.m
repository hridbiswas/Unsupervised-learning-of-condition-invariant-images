clear all;
close all;
clc;
im_dimension=[26,64];
matrix=[];
srcFiles = dir('F:\DATASETS\Alderly_Day_Night_DAtaset\for training\*.jpg');  
for j = 1 : length(srcFiles)
filename = strcat('F:\DATASETS\Alderly_Day_Night_DAtaset\for training\',srcFiles(j).name);
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