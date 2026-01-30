# Can Monte Carlo dropout help you to sample better examples for Active Learning?

In this work, I implement Monte Carlo dropout and use the highest entropy
of the prediction as a sampling strategy for Active Learning. I
compare the performance of this sampling strategy with other simple strategies,
such as the highest Variation Ratios, the lowest Margin and the
highest Entropy (without dropout).

I am using [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset for model training.
The rest should be narratively explained in the notebooks and/or in the attached report.

The project was created as an exam project for *Machine Learning for Natural Language Processing*
course in the *Cognitive Science* Master's program in the *University of Trento*.

Don't be shy to use my code, scientific community should help each other!

Papers I have used for my creation:
1. Gal, Y., Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning. In Proceedings of the
33rd International Conference on Machine Learning (pp. 1050-1059).
2. Suhan S., Junhee S. (2025). Improving Monte Carlo dropout uncertainty
estimation with stable output layers. In Neurocomputing (Vol. 661).
3. Yarin, G., Jiri, H., Alex, K.(2017). Concrete Dropout.
4. Korsch, D., Shadaydeh, M., Denzler, J. (2025). Simplified Concrete Dropout
\- Improving the Generation of Attribution Masks for Fine-grained Classification. In International Journal of Computer Vision 133 (pp. 5857–5871).
5. Ji, Y., Kaestner, D., Wirth, O., Wressnegger, C. (2023). Randomness Is
the Root of All Evil: More Reliable Evaluation of Deep Active Learning.
In Proceedings of the IEEE/CVF Winter Conference on Applications of
Computer Vision (pp. 3943-3952).
6. Edrina, G., Jiankang, D., Ismail, E. (2025). Deep active learning: A reality
check In Pattern Recognition Letters (Vol. 197, pp. 188-194).
7. Oscar, R., Abdulrahman, H. A., Sebastián, V. (2018). Statistical comparisons
of active learning strategies over multiple datasets In Knowledge-Based
Systems (Vol. 145, pp. 274-288).
8. Zhan, X., Wang, H., Huang, S. J., Chen, J. L., Wang, Y., Du, Y. (2022). A
Comparative Survey of Deep Active Learning
9. Asgharnezhad, H., Shamsi, A., Alizadehsani, R., Mohammadi, A., AlinejadRokny, H. (2025). Enhancing Monte Carlo Dropout Performance for Uncertainty Quantification.
