    img_name = "/data/Pytorch_Porjects/depthmap/result_3_x4/outdoor_005_haze/img_depth/depth_imwrite.png";
    scribs_img_name = "/data/Pytorch_Porjects/depthmap/result_3_x4/outdoor_005_haze/img_depth/depth_imwrite.png";
    I=double(imread(img_name))/255;
    mI=double(imread(scribs_img_name))/255;
    consts_map=sum(abs(I-mI),3)>0.001;
    
    [h,w,c]=size(I);
    N=w*h;
    
    Diag = spdiags(ones(N,1), 0, N, N);
    Ds=spdiags(consts_map(:),0,N,N);
    epsilon=20;
    lambda=4;
    %lambda越小效果与好
    win_size=3;
    L=getLaplacian1(I,consts_map,epsilon,win_size);
    I1=I(:,:,1);
    x=(L+lambda*Diag)\(lambda*I1(:));
    alpha=reshape(x,h,w);
    %figure, imshow(alpha);
    filename1 = "/data/Pytorch_Porjects/depthmap/result_3_x4/outdoor_005_haze/img_depth/mt_depth_imwrite";
    filename = [filename1, '_', num2str(epsilon), '_', num2str(lambda), '.png'];
    filename = strjoin(filename, '');
    disp(filename);
    imwrite(alpha, filename);
    %drawnow;