package com.example.myapplication;

import java.util.Arrays;
import java.util.Random;
import java.util.Arrays;
import java.util.Random;

public class HMM {
    private int numStates;
    private int numObservations;
    private double[][] transitionProb;
    private double[][] emissionProb;
    private double[] initialProb;

    public HMM(int numStates, int numObservations) {
        this.numStates = numStates;
        this.numObservations = numObservations;
        this.transitionProb = new double[numStates][numStates];
        this.emissionProb = new double[numStates][numObservations];
        this.initialProb = new double[numStates];

        initializeProbabilities();
    }

    private void initializeProbabilities() {
        Random random = new Random();

        // Initialize transition probabilities
        for (int i = 0; i < numStates; i++) {
            double sum = 0;
            for (int j = 0; j < numStates; j++) {
                transitionProb[i][j] = random.nextDouble();
                sum += transitionProb[i][j];
            }
            for (int j = 0; j < numStates; j++) {
                transitionProb[i][j] /= sum;
            }
        }

        // Initialize emission probabilities
        for (int i = 0; i < numStates; i++) {
            double sum = 0;
            for (int j = 0; j < numObservations; j++) {
                emissionProb[i][j] = random.nextDouble();
                sum += emissionProb[i][j];
            }
            for (int j = 0; j < numObservations; j++) {
                emissionProb[i][j] /= sum;
            }
        }

        // Initialize initial state probabilities
        double sum = 0;
        for (int i = 0; i < numStates; i++) {
            initialProb[i] = random.nextDouble();
            sum += initialProb[i];
        }
        for (int i = 0; i < numStates; i++) {
            initialProb[i] /= sum;
        }
    }

    public int getNumObservations() {
        return numObservations;
    }

    public void setTransitionProb(double[][] transitionProb) {
        this.transitionProb = transitionProb;
    }

    public void setInitialProb(double[] initialProb) {
        this.initialProb = initialProb;
    }

    public void setEmissionProb(double[][] emissionProb) {
        this.emissionProb = emissionProb;
    }

    public double[][] getTransitionProb() {
        return transitionProb;
    }

    public double[][] getEmissionProb() {
        return emissionProb;
    }

    public double[] getInitialProb() {
        return initialProb;
    }
}
