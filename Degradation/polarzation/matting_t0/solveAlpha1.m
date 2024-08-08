function alpha=solveAlpha1(I,consts_map,consts_vals,varargin)
  
  [h,w,c]=size(I);
  img_size=w*h;

  A=getLaplacian1(I,consts_map,varargin{:});
  Diag = spdiags(ones(img_size,1), 0, img_size, img_size);
  D=spdiags(consts_map(:),0,img_size,img_size);
  lambda=100;
  I1=I(:,:,1);
  x=(A+lambda*Diag)\(lambda*I1(:));
 
  %alpha=max(min(reshape(x,h,w),1),0);
  alpha=reshape(x,h,w);
