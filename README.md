# Kapellmeister
Mobile application for blind and low-vision people which leverages ML models to create new value. 

## Report

**Kapellmeister -- Multi-Objective Optimization of Container Allocation and Rescheduling with Genetic Algorithm**
[1] A. Velikanov, S. Torekhan, K. Baryktabasova, T. Mathews,  [Kapellmeister -- Multi-Objective Optimization of Container Allocation and Rescheduling with Genetic Algorithm](GenGenPaper.pdf)

## Developer guide
From Professor's document:
"Make sure your submission is self-contained. If it has any external dependency, either include it in the repository or provide a detailed instruction on how to install them. We expect your project to work out of the box with reasonable ease."<br/>
 "The repository should contain sufficient information so that we can execute your project." !!!!!!!!!!

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
