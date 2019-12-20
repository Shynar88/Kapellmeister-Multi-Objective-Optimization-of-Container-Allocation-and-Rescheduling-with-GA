# A-Eye
Mobile application for blind and low-vision people which leverages ML models to create new value. 

## Report

**A-Eye – leveraging ML models for helping low-vision and blind people**
[1] S. Torekhan, S. Woo, G. Lee, M. Ahmadli,  [A-Eye – leveraging ML models for helping low-vision and blind people](GenGenPaper.pdf)

## Demo
put GIFs of both server side and client side working

## Developer guide

**iOS application**<br/>
All the client side code is in the A-Eye folder.<br/>
**Dependencies** XCode, IPhone with iOS version >= 12.3.1, Alamofire(4.7.3), SwiftyJSON(4.0), AFNetworking(2.5.4)<br/>
Set up instructions:<br/>
Guide on [XCode installation](https://medium.com/@LondonAppBrewery/how-to-download-and-setup-xcode-10-for-ios-development-b63bed1865c). Note: Xcode can be installed only on MacOS<br/>
Guide on [how to run application on IPhone](https://codewithchris.com/deploy-your-app-on-an-iphone/)<br/>
Guide on [dependencies installation via cocoapods](https://www.raywenderlich.com/626-cocoapods-tutorial-for-swift-getting-started). The Podfile with dependencies is already provided. <br/>

**Server**<br/>
All server side code is in the server.ipynb file.<br/>
**Dependencies** Server code takes care of all dependencies installation.<br/>

**How to reproduce Demo results**<br/>
Run the first cell of Server.ipynb. It will take care of dependecies installations for all models and server environment. 
Run the seconde call of Server.ipynb. It will run the server. The ngrok address will be displayed as output. Copy the address(ex:123124ngrok.io) and paste it on line 123 of ViewController.swift. Build and run the Xcode Project on IPhone. Done. Note: don't run Xcode project on Xcode's simulator, it will not work, because access to IPhone's camera is needed. <br/>

## References 
Implementation of TTS inference code uses [TTS implementation by CorentinJ](https://github.com/CorentinJ/Real-Time-Voice-Cloning.git)<br/>
Implementation of Image Captioning inference code uses [Show and Tell: A Neural Image Caption Generator](https://github.com/tensorflow/models/tree/master/research/im2txt)<br/> and [Pretrained-Show-and-Tell-model](https://github.com/KranthiGV/Pretrained-Show-and-Tell-model)<br/>
Implementation of OCR inference code uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)<br/>
Implementation of Object Detection inference code uses [Facebook AI Research's Detectron2](https://github.com/facebookresearch/detectron2)<br/>

## Inquiries
If you have any difficulty with the reproduction of the results or any other questions, feel free to contact authors by email: shynartorekhan (at) gmail.com

