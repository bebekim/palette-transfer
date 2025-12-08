# Palette-based Color Transfer between Images

**Chenlei Lv**, College of Computer Science and Software Engineering, Shenzhen University, China  
**Dan Zhang**, School of Computer Science, Qinghai Normal University, China

## Abstract

As an important subtopic of image enhancement, color transfer aims to enhance the color scheme of a source image according to a reference one while preserving the semantic context. To implement color transfer, the palette-based color mapping framework was proposed. It is a classical solution that does not depend on complex semantic analysis to generate a new color scheme. However, the framework usually requires manual settings, reducing its practicality. The quality of traditional palette generation depends on the degree of color separation.

In this paper, we propose a new palette-based color transfer method that can automatically generate a new color scheme. With a redesigned palette-based clustering method, pixels can be classified into different segments according to color distribution with better applicability. By combining deep learning-based image segmentation and a new color mapping strategy, color transfer can be implemented on foreground and background parts independently while maintaining semantic consistency. The experimental results indicate that our method exhibits significant advantages over peer methods in terms of natural realism, color consistency, generality, and robustness.

## 1. Introduction

Following the development of digital media technology, researchers propose various enhanced frameworks to satisfy related requirements on digital display. Such frameworks include image denoising, super-resolution, low-light enhancement, color transfer, style transfer, etc. As an important subtopic, the color transfer framework attempts to learn the color scheme from the reference image to recolor the source one.

The pioneer work of color transfer was proposed by Reinhard in [5]. It establishes the relationship between images in Lab color space, which provides a decoupled representation of color information. The main drawback of the work is that the semantic correspondences between images are ignored.

To solve the problem, the palette-based color transfer methods [9] are proposed. They implement the color mapping between images based on color segments represented by a palette. Such a scheme has two significant advantages: independence from semantic correspondence and consistency in color distribution.

In this paper, we propose a new palette-based color transfer framework without any manual inputs. It is constructed by three components: palette-based clustering, color mapping strategy, and lighting optimization.

**Contributions:**
- We present a new palette-based clustering method to establish the palette for images. It improves the accuracy and applicability of palette generation. Compared to traditional solutions like the k-means clustering-based method, our method doesn't require specifying clustering centers. The palette can be generated automatically based on histogram analysis in the Lab color space.

- We propose a color mapping strategy to implement color transfer based on the achieved palette. The strategy aligns palette between images with precise chromatic aberration control. Combining a deep learning-based image segmentation, the color mapping can be implemented for foreground and background parts independently while keeping the color consistency.

- We provide a lighting optimization as an optional action for reference images with abnormal exposure. The optimization can be regarded as a global adjustment according to illuminations.

## 2. Related Works

According to different mechanisms of color transfer, the related works can be classified into three categories: global color transfer, correspondence-based color transfer, and deep feature-based color transfer.

**Global color transfer methods** attempt to establish color mapping according to global color distribution. The pioneer work is proposed by Reinhard [5] in 2002. It constructs the color mapping in Lab color space which provides a decoupled representation of color information.

**Correspondence-based color transfer methods** implement color mapping based on palette or semantic correspondence. The methods depend on semantic correspondences between images. Once the images for color transfer lack correspondences, the performance of the methods will drop significantly.

**Deep feature-based color transfer methods** represent the latest trend in the field. Benefited from the robust semantic analysis and feature learning on large dataset, more accurate color transfer result can be generated based on the trained deep neural network.

## 3. Palette-based Clustering

For palette-based color transfer or recoloring works, the most important issue is how to generate a palette to represent color distribution. One classical solution is to cluster colors into different categories with a user-specified cluster center number. However, different center numbers can lead to different palettes for the same image, resulting in unpredictable effects for color mapping.

An ideal palette generation scheme should consider various color distributions for different images and generate adaptive palettes. Similar colors should be classified into the same item of the palette while maintaining separation for different colors.

### 3.1 Algorithm Components

The clustering method establishes a palette for the image based on color histogram analysis in the Lab color space. The color histogram reflects aggregate information for different colors. The clustering method searches for peak values to generate a palette while aggregating related adjacent colors.

The searching process is implemented in the Lab color space because the color representations in the RGB space are coupled across three channels. Additionally, the influence of various lighting conditions cannot be conveniently removed. In the Lab color space, the color representations in different channels are decoupled, making it easier to search for accurate peak values and eliminate lighting effects.

The method includes three core parts: **histogram construction**, **peak searching**, and **peak merging**.

### 3.2 Histogram Construction

We divide each channel (l_i, a_i, b_i) of Lab into specified z bins and accumulate pixels into the related bin b_i (z = 100 by default). Then, a continuous color distribution is converted to a discrete form for different channels. This process is shown as:

```
I_H = {b_0, ..., b_z}, b_i = (l_i, a_i, b_i), i ∈ [0,z]
```

where I_H represents the histogram of input image I in Lab color space.

### 3.3 Peak Searching

We compare each bin b_i one by one and select the peak that has maximum value in a small range N{b_i} into the candidate peak set {b_p}. The formulation can be represented as:

```
{b_p} = max{b_i|N(b_i), b_i > b_min},
N{b_i} = {b_{i-r}, ..., b_{i+r}}
```

where r represents the searching radius, which defines the scale of local range N{b_i}. In practice, we set r to 3 as a default value. The peak searching is implemented in local range. Random color distributions may produce some fake peak values without statistical significance. To reduce the influence, we add a threshold-based control (b_i > b_min, b_min = 30 by default) to avoid adding some peak with few related pixels.

### 3.4 Peak Merging

Theoretically, we can compute the palette from the b_p directly. However, we have observed that the scale of the peak set b_p exceeds the acceptable range in most cases, resulting in a generated palette that is too large, thus reducing the aggregation property for colors.

To address this issue, we introduce a peak merging step to reduce the scale of b_p. We collect the achieved b_p from the three channels of the Lab color space. By combining the three channels, we generate one set of peak colors and construct a kd-tree from the set. Then, we input the Lab value of each pixel one by one into the kd-tree and count the number related to each peak.

An upper bound number t for the peak set can be specified. The final peak set b_f^p is computed by:

```
{b_f^p} = {b_i|K(b_i) > K(b_t), b_i ∈ {b_p}}
```

where K represents the pixel number accumulation for related bin b_i by kd-tree searching, b_t is the bin that the order of K(b_t) is same to t in K{b_p}.

Once the final peak set {b_f^p} is achieved, all pixels can be classified based on the set. The classification is shown as:

```
p_i = arg min d(L(p_i), b_i), b_i ∈ {b_f^p}
```

where p_i represents a pixel of the input image I, L(p_i) represents the Lab values of p_i, and d is the distance between L(p_i) and different peak values in b_f^p. The classification result of p_i is the index of b_i which has smaller value of d in Lab color space.

The palette can be generated based on the average values in related categories. The selection of upper bound number t controls the scale of final peak set that decides the length of palette. To balance the aggregation property and scale control of palette, we set t to 32 based on experience.

## 4. Color Mapping Strategy

Based on the palette-based clustering, the color transfer task can be conveniently transformed to find color mapping f, f: I_s{b_f^p} → I_r{b_f^p}. I_s represents the source image and I_r represents the reference one.

We propose three constraints to build the mapping f, including **split correspondence**, **chromatic aberration control**, and **color consistency keeping**.

### 4.1 Split Correspondence

The split correspondence means that the mapping is implemented for the foreground and background parts independently. For most images, the internal content can be divided into two parts: foreground and background. The foreground typically represents semantic objects such as people, animals, and buildings, while the background serves as a context or backdrop to highlight the foreground objects.

It is natural to implement color mapping for the foreground and background separately. With advancements in deep learning frameworks, the separation between foreground and background parts can be accurately achieved. In our framework, we utilize DeepLabV3+ [12] to achieve this separation.

Based on this separation, the color mapping function f is divided into two parts:

```
f = {
  f_fore : I_s{b_f^p}^fore → I_r{b_f^p}^fore
  f_back : I_s{b_f^p}^back → I_r{b_f^p}^back
}
```

where f_fore and f_back represent the color mapping for the foreground and background parts, {b_f^p}^fore and {b_f^p}^back represent palette-based categories belong to the related parts.

The split correspondence keeps the basic correspondence for foreground and background color distributions. The split correspondence is independent to the specific semantic information between images. It doesn't require the strict semantic correspondence to implement color mapping.

### 4.2 Chromatic Aberration Control

The chromatic aberration control means that the color mapping between different peak values of related palettes should be controlled in a reasonable range. For instance, a random mapping may change colors of regions with high light representation to low one. Such mapping breaks light intensity-based semantic representation.

To control the mapping, we use the nearest neighbor searching to find the b_j from I_r{b_f^p} for b_i, represented as:

```
f_i: b_i → b_j, b_j = arg min d(b_i, b_j)
```

where d is the distance in Lab color space. Based on the nearest neighbor searching, the basic information of chromatic aberration can be inherited from source image to transfer result while considering the color scheme of reference one.

Although the nearest neighbor searching strategy for chromatic aberration control reduces the flexibility of color mapping, it avoids disorder color transfer. The property is important for the generality in practice.

### 4.3 Color Consistency Keeping

The color consistency keeping aims to maintain the basic structure of color distribution during the color mapping process. This method consists of **internal consistency keeping** and **external continuity keeping**.

In the part of chromatic aberration control, we discussed the use of nearest neighbor searching to map peak values from the reference image to the source image. However, this searching strategy does not guarantee a one-to-one mapping, meaning that some peak values of the source image may share the same peak value from the reference image. When a many-to-one mapping is established, the structure of color distribution is inevitably disrupted.

#### Internal Consistency Keeping

To solve the problem, the internal consistency keeping is proposed. The related formulation is represented as:

```
f_k : b_k → b_j, b_k = max{b_i|b_i ∈ {b_I}}
```

where f_k is used to instead f_I and b_k is selected from {b_I} with the maximum number of pixels. Other peak values in {b_I} are added into the pending set {b_Q}. Then, the many-to-one mapping is eliminated.

#### External Continuity Keeping

The external continuity keeping is to finish the color mapping for the set {b_Q}. The mapping values of {b_Q} are computed based on existing mapping results with corresponding weights. The mapping can be formulated as:

```
f_q : b_q → Σ_{b_k∈N(b_q)} w_k f(b_k), b_q ∈ {b_Q}

w_k = d(b_k, b_q)^{-1} / Σ_{b_k∈N(b_q)} d(b_k, b_q)^{-1}
```

where b_p is a peak value of {b_P}, b_k represents the peak value which has obtained mapping value, N(b_p) is a neighbor set of b_P. The mapping value of b_p is computed by its neighbor's mapping. The related weight is computed by the reciprocal of distance d.

### 4.4 Final Color Transfer

For each pixel in source image, the related color transfer result can be computed by:

```
f(p_i) = f(b_i) + L(p_i) - b_i
```

where p_i is a pixel in source image I_s, b_p is the peak value related to the p in palette, L(p) is the Lab values of p, p'_i is the related transfer result, p'_i = f(p_i).

Based on the color mapping result f(p_i), the new Lab-based values L(p'_i)_a and L(p'_i)_b in a and b channels of p'_i can be achieved by L(p'_i)_a = f(p_i)_a, L(p'_i)_b = f(p_i)_b. The value L(p')_l of L channel of p is not changed in practice. We update L(p')_l in lighting optimization.

## 5. Lighting Optimization

After color mapping, the values of the a and b channels for the source image have been transferred according to those of the reference image. However, the values in the L channel are processed individually from the mapping. The reason is that such values reflect the lighting intensity of the pixels.

The values in the L channel should be optimized separately in an independent step. We design a lighting optimization to be the step. It is constructed by two part: **color mapping-based weighted update** and **global lighting enhancement**.

### 5.1 Color Mapping-based Weighted Update

The color mapping-based weighted update is to add mapping result from f(p_i) to the original L channel with a specified weight, represented as:

```
L(p'_i)_l = (1 - α)L(p_i)_l + α f(p_i)_l
```

where α is used to control the weight of f(p_i)_l that is the color mapping result in L channel. In practice, we set α to 0.3 by default.

### 5.2 Global Lighting Enhancement

For some reference images with abnormal exposure regions, we optionally employ the global lighting enhancement to optimize the values in the L channel. The basic framework of this enhancement is implemented according to [43]. We extract the values of the L channel processed by the enhancement and reassign them as L(p')_l. This process helps improve the quality of transfer results with abnormal exposure regions produced by the reference image.

## 6. Experiments

### 6.1 Quantitative Analysis

The quantitative analysis for color transfer is still an open problem. We design some analysis tools to build stable quantitative analysis for color transfer. The tools include **color consistency analysis** and **fading rate computation**.

#### Color Consistency Analysis

The color consistency analysis is used to evaluate the degree of color deviation after color transfer. Some pixels with approximate RGB and L values in source image should keep the similarity in transfer result. To quantify the evaluation, we compute the histograms for RGB (10 bins for each main color) and L (20 bins) channels from source image. Based on the histograms, the pixels can be classified into related bins. Then, we calculate the variance in each bin with corresponding RGB and L values from transfer result. The average value of variances can be used to represent quantitative result of color consistency analysis.

#### Fading Rate Computation

The main purpose of color transfer is to improve the quality of color distribution in source image. It can be regarded as a kind of image quality enhancement. If we consider the color distribution as signals, the strength of the signals should not be weakened by color transfer.

Based on the principle, we present the fading rate computation to quantitative represent the degree of color-based signal loss. We extract a and b values in Lab color space from source image and transfer result. The values are regarded as the color-based signals. We compute the difference of the values pixel by pixel between source image and transfer result to be the signal loss (when the value in transfer result is larger than source one, the signal loss is set to 0). Finally, we calculate the average from the signal loss to be the fading rate.

### 6.2 Results and Comparisons

The experimental results indicate that our method exhibits significant advantages over peer methods in terms of natural realism, color consistency, generality, and robustness. Our method achieves better performance in color consistency keeping and generates more stable and accurate color transfer results.

Compared to traditional palette-based color transfer or recoloring methods, our framework doesn't require manual input. The proposed framework provides a reasonable solution for images without strict semantic correspondence requirements. It avoids discontinuity in color distribution and unnatural color mapping as much as possible.

## 7. Conclusions

In this paper, we propose a palette-based color transfer framework. It provides an applicable palette-based clustering to aggregate discrete color distributions by histogram analysis in Lab color space. Based on the palette, the proposed color mapping strategy utilizes split correspondence, chromatic aberration control, and color consistency keeping to achieve more accurate and robust transfer result.

The experimental results show that our framework achieves better transfer performance based on various evaluations. In future work, we will improve the efficiency of the method and introduce structural information analysis to inhibit transfer degradation.

## Key Implementation Details

### Parameters
- **z**: Number of histogram bins per Lab channel (default: 100)
- **r**: Peak searching radius (default: 3)
- **b_min**: Minimum peak threshold (default: 30)
- **t**: Upper bound for palette size (default: 32)
- **α**: Lighting optimization weight (default: 0.3)

### Processing Pipeline
1. **Input**: Source image, Reference image
2. **Palette Extraction**: Histogram analysis → Peak detection → Peak merging
3. **Semantic Segmentation**: DeepLabV3+ for foreground/background separation
4. **Color Mapping**: Split correspondence + Chromatic aberration control + Color consistency
5. **Lighting Optimization**: Weighted L-channel update + Optional global enhancement
6. **Output**: Color-transferred image

### Key Advantages
- **Automatic**: No manual parameter tuning required
- **Semantic-aware**: Separate foreground/background processing
- **Robust**: Multiple constraints prevent artifacts
- **Generalizable**: Works without strict semantic correspondence
