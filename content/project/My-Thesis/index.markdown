---
authors:
- admin
date: "2019-09-23T00:00:00-03:00"
external_link: ""
image:
  caption: Photo by [Charlie Foster](https://unsplash.com/@charliefoster) on Unsplash
  focal_point: Smart
links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/tobar_with_R
#slides: example-slides

summary: This is What my Thesis Project will be about.
tags:
- Civil Engineering
- Finite Elements
- Stiffness Method

title: My Thesis
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
---

Coming up with an Interesting Thesis Project is not easy at all. Actually I had 3 different projets and 5 different professors. None of them were really interested in my propositions. Thank God I found [Dr. Marcos Valdebenito](http://www.ociv.usm.cl/profesores/valdebenito-c-marcos-2/). He is really interested in Reliability Analysis in Structures and Study the Response of Random Field Variables into Strutures, you can learn more about his work on his website. Once I talked to him about I was doing in my former job he was really interested in applying Machine Learning techniques to solve this kind of problems.

Normally to study the effect of Ramdom fields in Structures a Montecarlo Simulation is run several times to determine how the Structure response is affected. This is done by analyzing the mean and the Covariance of the simulations. This process is omputationally expensive since normally 10,000 to 1,000,000 simulations are needed. Every one of those simulations solves the following problem, also called the Rayleigh - Ritz Method:

$$ [K] \{u\} = \{f\}$$

Where $ [K] $ is the Finite Element Matrix representing the Equivalent Stiffness of the different Degrees of Fredom of the Structure.  $ \{f\} $ is the Equivalent Load Vector representing the forces affecting the Structure. In order to solve this problem $ [K]^{-1} $ needs to be pre-multiplied with $ \{f\}$ to obtain the Struture Response $ \{u\} $ representing the Structure displacement at every Degree of freedom. Normally $ [K] $ is a fairly large Matrix and the Inversion process costly so alternative methods to Montecarlo Simulation are deeply appreciated.

So that is how we came up with a Project. What about considering $ [K] $ as a black and white image (1 channel), representing the stiffness of a Truss. Therefore, Convolutional Networks could be a good alternative to analyze the Matrix and train a Network capable of determinimg in a first instance the displacement of the Structure ($\{u\}$) and afterwards the failure of the Structure, transforming the problem into a Clasification Binary Problem (Failing - not Failing).

I'll be posting more technical content about how I've been tackling the problem. Stay tuned!!!

