# JPEG compressed image detection and estimation

This report was prepared for a home assignment for a job application. Here I will give the content and Python scripts to run the algorithm.

## 1 Problem Statement
### 1.1 Research problem: JPEG compressed image detection and estimation
"We are interested in automatically detecting whether images were jpeg compressed or not. And estimation of compression level (aka quality factor) if possible. Jpeg compression works by splitting images into non-overlapping 8x8 blocks, transforming them into a DCT domain, and then quantizing high-frequency coefficients, thus reducing the image in size. Jpeg compression is usually applied multiple times (with block shifts) on real-world images because they are re-uploaded to different internet resources. This makes them quite messy and causes artifacts on the output of our networks. The
path towards automatic classification and jpeg factor estimation might be long, so perhaps some visual examination solution would be interesting to start with, where some graphic diagram or something else can clearly testify that the whole image or part of it was jpeg compressed. Such a solution which has the potential to be further developed into an automatic detector/estimator probably might be enough. We prefer not to have neural networks or other relatively black-box solutions. We would like to see explainable analytic approaches. But it's not totally mandatory, but rather a wish.
In production, we usually need to just solve the problem. It's OK to use everything. If you use an approach from a certain paper, please cite it here and be able to explain it to us. We provided a set of images which are quite typical cases for us. Some of them are cleaner, some are artificially jpeg corrupted and some come from the real world. If you can, you can demo your solution using them for us."

### 1.2 Failed Approach
First, I visually examined the uncompressed and compressed images and analyzed the block artifacts of JPEG compression. First I thought using a median filter would reduce the artifacts. The median filter is a well-known denoising and smoothing operator. It is especially effective for removing impulses. It is invariant to sustained changes in signal intensity. First I converted the RGB image to YCbCr and used only the luminance channel. Then I used a 1x8 median filter to enhance the artifacts by subtracting the median filtering result from the compressed image's luminance channel.
After this subtraction, I summed up the rows of the image. The results are given in Figure 1 and Figure 2. When the image is compressed with the JPEG algorithm, the block artifacts line up and the summation of these enhanced the artifacts and removes most of the other image content. We see impulses that are separated 8 pixels apart (Figure 1a 3rd row). Because the Fourier transform of an impulse train is also an impulse train, at the FFT output we should see peaks that are at exact locations of M/8 where M is the length of FFT. I marked those locations in Figure 1 and Figure 2
with square markers in magenta color. When the image is highly compressed, we should see more block artifacts, thus the FFT peak values should be higher. The uncompressed image has no such artifacts so we do not see any peaks in the FFT output. I did not continue with this method because this method is also affected by the image content and size. However, this method can be improved if it can be worked on a little more.

### 1.3 Working Approach
After the first attempt, I did some more research on the JPEG compression algorithm and discrete cosine transform (DCT). I decided to go with analyzing the DCT coefficients. Specifically, I used the DC component of DCT coefficients since it is not affected by the image content. The details of the method are given in the next section.

![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/dffdf02f-c550-41af-8c2f-d05579e318ee)


![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/c39e8cbc-44eb-4972-bb92-ca858e354595)


## 2 JPEG compressed image detection and estimation
JPEG compression uses discrete cosine transform (DCT), quantization, and entropy coding. After these steps, the data is saved to a file. At the decoding phase, the previous steps are reversed. The DCT is done on 8x8 non-overlapping blocks. In Figure 3, an uncompressed image and its compressed versions are shown. If the compression quality is very low (high compression) such as 10, then the so-called blocking artifacts is clearly
visible. These blocking artifacts are pixel value discontinuities across block boundaries.

![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/e1b1dd0b-5bae-470b-8f3d-af1942082faf)


First, I've converted the RGB image to YCbCr and used only the luminance channel in the next steps. Then, I extracted the DCT coefficients of the uncompressed and compressed images. These are plotted in Figure 4. In the second row, the DCT coefficients are shown for only a single 8x8 block. The first digit, at the top left of the 8x8 block, of DCT coefficients is called the DC (direct current) component. The rest 63 components are called AC (alternate current) components. It looks like when the image is uncompressed, The AC components are mostly non-zero. When it is compressed, the high-frequency parts in the AC components are rounded to zero. Also, I expect that the quantization operation on DCT coefficients force the value of each DCT coefficient to be an integer multiple of the quantization step size.

![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/fa8a69b3-2807-476e-a541-3f12aa391fcb)

It was reported that the DC component has a Gaussian distribution [1]. In Figure 5 I have plotted the histogram of the DC component and observed that the DC histogram for the uncompressed image actually looks like a Gaussian distribution. The DC histogram of the compressed images are clustered around integer multiples of the quantization matrix [2]. When the compression quality is lower, the cluster centers of DC components are highly separated. So actually the distance between the clusters are directly correlated with the compression rate. To understand if an image is JPEG compressed or not, and to know how much the image is compressed, I decided to analyze the DC histogram plots. Measuring the distance between the clusters in the DC histogram sounds logical to me. To do that, I needed to measure the periodicity of the clusters. The histograms of DC component when the image is compressed look like an impulse train. The Fourier transform of a periodic impulse train in time is also a periodic impulse train in frequency. So I used FFT and in Figure 6 the outputs for three histograms are given. When the image is uncompressed, there is only the DC component of FFT at the zero frequency. When the image is compressed, the peaks of the impulse train are clearly visible. The distances between each peak is inversely correlated with the compression rate. I wrote a script to measure the distances of peaks in FFT output. Let's call this distance K = d/N ∗ 100 where d is the distance between peaks in Hz and N is the sampling frequency. It is a normalized term and so not affected by the changes in image size. When the algorithm can not find any peak at FFT output, other than the DC component of FFT, K is set to 100.

![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/618d6d52-4b44-422e-bfdb-006958b70ec3)

I downloaded the train data from [here](https://data.vision.ee.ethz.ch/cvl/DIV2K/). There are 100 uncompressed images in PNG file format. First I have resized all to 256x256 and saved with different JPEQ qualities of q = [10, 20, 30, 40, 50, 60, 70, 90, 100]. q = 100 means there is no compression, and q = 10 means it is highly compressed. Then for each one, I have extracted the K values. The mean result for each compression quality is given in Figure 7. Here I linearly interpolated the intermediate values and created a lookup table. So in the experiments, I use this lookup table to detect the JPEG compression quality.

![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/60e19c57-53db-4a78-905f-358b15ade2a8)

![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/8ace4c5a-597a-4439-bc49-f91ff83f44b0)


## 3 Validation Data Results
Here I give the validation data results in terms of visual outputs and detected JPEG quality. I assume that if any image is compressed with the JPEG algorithm, then the 8x8 blocks start from the (0, 0) pixel in the image, which means I am assuming there is no grid alignment problem. I also
assume the compressed images are only compressed once, thus ignoring any issues with the double JPEG compression problem.

**Sample-1**
![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/583a64d8-8a4e-4336-96dd-e4af81d22373)

**Sample-2**
![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/f6858714-31ce-4058-828c-9845238c1afa)

**Sample-3**
![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/cf1bb3f7-441d-4023-97dc-e43bd5d98cd2)

**Sample-4**
![image](https://github.com/himmetozcan/jpeg_compression/assets/44242024/5a44e1cd-616e-467e-9787-5f3a28bda31c)



## References

**[1]** Z. Fan and R. L. De Queiroz, Identification of bitmap compression history: Jpeg detection and quantizer estimation, IEEE Transactions on Image Processing, vol. 12, no. 2, pp. 230-235, 2003.
**[2]** M. C. Stamm, S. K. Tjoa, W. S. Lin, and K. R. Liu, Anti-forensics of jpeg compression, in 2010 IEEE
International conference on acoustics, speech, and signal processing. IEEE, 2010, pp. 1694-1697.

