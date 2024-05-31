Problem Statement

Developing a satellite image denoising algorithm targeting speckle noise removal, ensuring clarity enhancement for applications like environmental monitoring. It should be robust against speckle noise, preserving crucial details.

Introduction

In satellite imaging, it's crucial to have clear and accurate pictures for tasks like keeping an eye on the environment, planning cities, and managing disasters. But sometimes, these pictures get distorted by unwanted stuff like speckle noise. To fix this, we've made a smart tool called a denoising algorithm. It cleans up the images, making them clearer and better for understanding. This tool is really good at dealing with speckle noise, the most common issue in satellite images. It knows how to get rid of the noise while keeping the important details safe. It's perfect for quickly working through lots of satellite pictures. And it's set up to work smoothly on cloud systems, meaning it can process images in real-time without any hassle. This makes it super easy for people to get the cleaned-up pictures they need, exactly when they need them, for making smart decisions.

Dataset

We have created our own dataset by downloading images from the Sentinel-1 satellite developed by the European Space Agency (ESA). Sentinel-1 is a Synthetic Aperture Radar (SAR) satellite developed by the European Space Agency (ESA) as part of the Copernicus program. It provides high resolution radar imaging of the Earth's surface, regardless of weather conditions and time of day. The data products include Level-1, Level-2, and Level-3 products, which are used for a wide range of applications such as land and ocean monitoring, disaster management, climate change, and scientific research. The Copernicus Open Access Hub provides free access to Sentinel-1 data products, which are invaluable tools for decision-making and policy-making. Sentinel-1 data products enable scientists, researchers, and decision-makers to gain a better understanding of our planet and the challenges we face.


Method - Denoising Autoencoder 

Denoising autoencoders are an extension of the basic autoencoders architecture. An autoencoder neural network tries to reconstruct images from hidden code space. In denoising autoencoders, we are introducing speckle noise to the images. The denoising autoencoder network will try to reconstruct the images. But before that, it will have to cancel out the noise from the input image data. In doing so, the autoencoder network will learn to capture all the important features of the data.

