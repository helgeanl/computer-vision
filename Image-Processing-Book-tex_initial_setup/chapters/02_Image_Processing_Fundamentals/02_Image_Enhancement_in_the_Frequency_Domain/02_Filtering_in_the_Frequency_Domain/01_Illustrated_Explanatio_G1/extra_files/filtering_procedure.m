% TDT4265, Assignment D1.2 
% - Example of using a lowpass gaussian filter to smooth an image
% Group 1
% - Khan, Shahrukh; 
% - Langåker, Helge-André; 
% - Schiager, Per Johannes Graatrud

%% Import Matlab example picture of a cameraman
f = imread('cameraman.tif');

%% Add padding to the image
M = size(f,1);
N = size(f,2);
P = 2*M;
Q = 2*N;
f_p = padarray(f,[M N],'post');

%% Center the transform, multiply by (-1)^(x+y)
for i = 1:P
    for j = 1:Q
        f_cp(i,j) = f_p(i,j)*(-1)^(i+j);
    end
end

%% Fourier transform
F = fft2(double(f_cp));

%% Gaussian filter of size PxQ
H = fspecial('gaussian',[P Q],50);

%% Multiply
G = H.*F;

%% Inverse fast fourier transform, using only the real parts
g_cp = real(ifft2(Gcp));

%% Multiply by (-1)^(x+y)
for i = 1:P
    for j = 1:Q
        g_cp(i,j) = g_cp(i,j)*(-1)^(i+j);
    end
end

%% Remove the padding from the image, leaving an MxN image
g = g_cp(1:M,1:N);

%% Show images of each step
figure(8)
subplot(3,3,1)
    imshow(f,[])
    title('a) Original image of size MxN')
subplot(3,3,2)
    imshow(f_p,[])
    title('b) Padded image of size PxQ')
subplot(3,3,3)
    imshow(f_cp,[])
    title('c) Image multiplied by (-1)^{(x+y)}')
subplot(3,3,4)
    imshow(log(abs(F_cp)),[])
    title('d) Image spectrum of F_p')
subplot(3,3,5)
    imshow(H_p,[])
    title('e) Gaussian filter kernel H')
subplot(3,3,6)
    imshow(log(abs(Gcp)),[])
    title('f) Spectrum of the product G = HF_p')
subplot(3,3,7)
    imshow(g_cp,[])
    title('g) Filtered image g_p(x,y)')
subplot(3,3,8)
    imshow(g,[])
    title('h) Cropped image g(x,y)of size MxN')
    