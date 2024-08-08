% 精细化程度主要取决于epsilon的大小
for i=26:48
    img_name = sprintf('/data/Pytorch_Porjects/depthmap/0_result/%03d/img_depth/depth_imwrite.png',i);
    scribs_img_name=sprintf('/data/Pytorch_Porjects/depthmap/0_result/%03d/img_depth/depth_imwrite.png',i);
    %img_name='/data/Pytorch_Porjects/depthmap/result_mianfen2/matting/block_5.png';
    %scribs_img_name='/data/Pytorch_Porjects/depthmap/result_mianfen2/matting/block_5.png';
    I=double(imread(img_name))/255;
    mI=double(imread(scribs_img_name))/255;
    consts_map=sum(abs(I-mI),3)>0.001;
    
    [h,w,c]=size(I);
    N=w*h;
    
    Diag = spdiags(ones(N,1), 0, N, N);
    Ds=spdiags(consts_map(:),0,N,N);
    epsilon=20;
    lambda=4;
    win_size=3;
    L=getLaplacian1(I,consts_map,epsilon,win_size);
    I1=I(:,:,1);
    x=(L+lambda*Diag)\(lambda*I1(:));
    alpha=reshape(x,h,w);
    
    %figure, imshow(alpha);
    filename1 = sprintf("/data/Pytorch_Porjects/depthmap/0_result/%03d/img_depth/mt_depth_imwrite",i);
    filename = [filename1, '_', num2str(epsilon), '_', num2str(lambda), '.png'];
    filename = strjoin(filename, '');
    disp(filename);
    %filename = '/data/Pytorch_Porjects/depthmap/result_mianfen2/matting/mt_img5_5.png';
    imwrite(alpha, filename);
    %drawnow;
end