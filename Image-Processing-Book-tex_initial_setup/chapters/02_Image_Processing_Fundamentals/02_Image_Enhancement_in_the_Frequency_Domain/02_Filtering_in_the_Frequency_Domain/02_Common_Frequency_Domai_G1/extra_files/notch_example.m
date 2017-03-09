% 1 Compute the fourier transform F of the image
f = imread('clown.tif');
figure, imshow(f, []);
F=fftshift(fft2(double(f)));
S=log(abs(F));
imwrite(S/max(S(:)), 'mask.tif');
pause;

% 2 Create the mask H
% Edit the mask.tif file in e.g. Microsoft Paint, draw black circles etc. at spikes, make sure to put them symmetrically. Save the file, then continue the program

H = imread('mask.tif');
H = H(:,:,1); % Paint saves a 3ch file, we extract one channel
H = double((H>0)); % Tresholding to finish the mask

% 3 Point-multiply the two spectra
G = H .* F;

% 4 Compute the inverse transform of the shifted product
g = real(ifft2(ifftshift(G)));
imwrite(g/max(g(:)), 'clown_result.tif');