Hybrid Approach: CNN‑DCT Steganography
In this paper, the study proposed a hybrid approach com-
bining CNNs and DCT for image steganography. The CNN
component will handle the embedding and extraction of hid-
den information within images, while DCT will be used for
the spatial domain steganography process, ensuring efficient
and secure data hiding (Fig. 1).
The block diagram shows the flow of the proposed hybrid
approach: CNN-DCT Steganography. Each node represents
a step in the methodology, and the arrows indicate the
sequence of operations between the steps.
1. Cover Image: The original image is taken as input to
the proposed hybrid steganography model.
2. Feature Extraction (CNN): The cover image is pro-
cessed by the CNN component to extract hierarchical
features. The CNN learns to identify important patterns
and features in the image.
3. Secret Data Embedding (DCT): The extracted fea-
tures from the CNN are divided into smaller blocks or
patches. DCT is applied to these blocks to convert them
into the frequency domain. The secret data (message
or another image) is then hidden in the high-frequency
components of the DCT coefficients.
4. Stego Image: The blocks with the embedded secret data
are combined to form the stego image, which contains
the concealed information.
5. Stego Image Transmission (if applicable): The stego
image can be transmitted over the network or stored in
a cloud-based repository.
6. Stego Image Receiving (if applicable): In the case of
cloud-based steganography, the stego image is received
from the cloud.
7. Secret Data Extraction (DCT Inverse): The stego
image undergoes DCT inverse to retrieve the embedded
secret data.
8. Feature Extraction (CNN): The retrieved secret data is
then processed by the CNN component again to extract
its hierarchical features.
9. Data Verification: The extracted data is verified and
compared with the original secret data to ensure accu-
racy and integrity