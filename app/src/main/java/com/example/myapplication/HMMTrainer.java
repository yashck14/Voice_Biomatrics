package com.example.myapplication;
public class HMMTrainer {
    private HMM hmm;

    public HMMTrainer(HMM hmm) {
        this.hmm = hmm;
    }

    public void train(int[] observations, int maxIterations) {
        int numStates = hmm.getInitialProb().length;
        int numObservations = hmm.getNumObservations();
        int T = observations.length;

        System.out.println("Number of States: " + numStates);
        System.out.println("Number of Observations: " + numObservations);
        System.out.println("Length of Observation Sequence: " + T);

        double[][] transitionProb = hmm.getTransitionProb();
        double[][] emissionProb = hmm.getEmissionProb();
        double[] initialProb = hmm.getInitialProb();
        double[][] alpha;
        double[][] beta;
        double[][] gamma;
        double[][][] xi;

        for (int it = 0; it < maxIterations; it++) {
            alpha = forward(observations, numStates, T, transitionProb, emissionProb, initialProb);
            beta = backward(observations, numStates, T, transitionProb, emissionProb);

            gamma = new double[T][numStates];
            xi = new double[T - 1][numStates][numStates];

            for (int t = 0; t < T - 1; t++) {
                double sumGamma = 0;
                for (int i = 0; i < numStates; i++) {
                    gamma[t][i] = alpha[t][i] * beta[t][i];
                    sumGamma += gamma[t][i];
                }
                for (int i = 0; i < numStates; i++) {
                    gamma[t][i] /= sumGamma;
                }

                for (int i = 0; i < numStates; i++) {
                    for (int j = 0; j < numStates; j++) {
                        xi[t][i][j] = alpha[t][i] * transitionProb[i][j] * emissionProb[j][observations[t + 1]] * beta[t + 1][j];
                        xi[t][i][j] /= sumGamma;
                    }
                }
            }

            double[] gammaSum = new double[numStates];
            for (int i = 0; i < numStates; i++) {
                gammaSum[i] = 0;
                for (int t = 0; t < T - 1; t++) {
                    gammaSum[i] += gamma[t][i];
                }
            }

            for (int i = 0; i < numStates; i++) {
                initialProb[i] = gamma[0][i]+0.135;
            }

            for (int i = 0; i < numStates; i++) {
                for (int j = 0; j < numStates; j++) {
                    double num = 0;
                    for (int t = 0; t < T - 1; t++) {
                        num += xi[t][i][j];
                    }
                    transitionProb[i][j] = num / gammaSum[i]+0.135;
                }
            }

            for (int j = 0; j < numStates; j++) {
                for (int k = 0; k < numObservations; k++) {
                    double num = 0;
                    for (int t = 0; t < T; t++) {
                        if (observations[t] == k) {
                            num += gamma[t][j];
                        }
                    }
                    emissionProb[j][k] = num / gammaSum[j]+0.135;
                }
            }
        }
        System.out.println("\n after processing transitionProb:-\n");
        for (int i = 0;i<5;i++){
            for (int j =0;j<5;j++){
                System.out.print(transitionProb[i][j]);
            }
        }System.out.println("\nafter processing emissionProb:-\n");
        for (int i = 0;i<5;i++){
            for (int j =0;j<27;j++){
                System.out.print(emissionProb[i][j]);
            }
        }
        System.out.println("\nafter processing OBSERVATIONS:-\n");
        for (int i =0;i< 5;i++){
            System.out.print(initialProb[i]);
        }
        hmm.setEmissionProb(emissionProb);
        hmm.setTransitionProb(transitionProb);
        hmm.setInitialProb(initialProb);
    }

    private double[][] forward(int[] observations, int numStates, int T, double[][] transitionProb, double[][] emissionProb, double[] initialProb) {
        double[][] alpha = new double[T][numStates];

        // Initialization
        for (int i = 0; i < numStates; i++) {
            alpha[0][i] = initialProb[i] * emissionProb[i][observations[0]];
            if (Double.isNaN(alpha[0][i])) {
                alpha[0][i] = 0;
            }
        }

        // Forward procedure
        for (int t = 1; t < T; t++) {
            for (int i = 0; i < numStates; i++) {
                alpha[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    alpha[t][i] += alpha[t - 1][j] * transitionProb[j][i];
                }
                alpha[t][i] *= emissionProb[i][observations[t]];
                if (Double.isNaN(alpha[t][i])) {
                    alpha[t][i] = 0;
                }
            }
        }

        return alpha;
    }


    private double[][] backward(int[] observations, int numStates, int T, double[][] transitionProb, double[][] emissionProb) {
        double[][] beta = new double[T][numStates];

        // Initialization
        for (int i = 0; i < numStates; i++) {
            beta[T - 1][i] = 1;
        }

        // Backward procedure
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < numStates; i++) {
                beta[t][i] = 0;
                for (int j = 0; j < numStates; j++) {
                    beta[t][i] += beta[t + 1][j] * transitionProb[i][j] * emissionProb[j][observations[t + 1]];
                    if (Double.isNaN(beta[t][i])) {
                        beta[t][i] = 0;
                    }
                }
            }
        }

        return beta;
    }

}
