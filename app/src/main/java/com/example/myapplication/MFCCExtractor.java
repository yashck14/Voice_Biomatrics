package com.example.myapplication;


import android.util.Log;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.io.TarsosDSPAudioFormat;
import be.tarsos.dsp.io.UniversalAudioInputStream;
import be.tarsos.dsp.mfcc.MFCC;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.util.fft.FFT;
import be.tarsos.dsp.util.fft.HammingWindow;
import weka.core.Attribute;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.io.FileWriter;
import java.io.IOException;




public class MFCCExtractor {
    static List<double[]> mfccFeaturesList = new ArrayList<>();
    static List<double[]> lpcFeaturesList = new ArrayList<>();
    static List<double[]> spectralFeaturesList = new ArrayList<>();

    public static void feature(String audioFile){
        System.out.println("\n");
        mfccFeaturesList.clear();
        lpcFeaturesList.clear();
        spectralFeaturesList.clear();
        extractFeatures(audioFile);
    }

    public static void extractFeatures(String audioFilePath) {

        try {
            // Load the audio file
            File file = new File(audioFilePath);
            FileInputStream fileInputStream = new FileInputStream(file);
            TarsosDSPAudioFormat format = new TarsosDSPAudioFormat(44100, 16, 2, true, false);
            UniversalAudioInputStream audioStream = new UniversalAudioInputStream(fileInputStream, format);
            AudioDispatcher dispatcher = new AudioDispatcher(audioStream, 1024, 512);

            // Initialize LPC parameters
            int lpcOrder = 12; // Order of the LPC analysis
            int sampleRate = 16000;
            int windowSize = 1024;
            HammingWindow hammingWindow = new HammingWindow();
            FFT fft = new FFT(windowSize, hammingWindow);

            // LPC coefficients buffer
            double[] lpcCoeffs = new double[lpcOrder];

            // LPC processor
            dispatcher.addAudioProcessor(new AudioProcessor() {
                @Override
                public boolean process(AudioEvent audioEvent) {
                    // Convert audio buffer to doubles
                    float[] audioBuffer = audioEvent.getFloatBuffer();

                    // Apply windowing
                    hammingWindow.apply(audioBuffer);

                    // Autocorrelation
                    double[] r = new double[lpcOrder + 1];
                    for (int i = 0; i <= lpcOrder; i++) {
                        for (int j = 0; j < windowSize - i; j++) {
                            r[i] += audioBuffer[j] * audioBuffer[j + i];
                        }
                    }

                    // Levinson-Durbin recursion
                    double[] k = new double[lpcOrder + 1];
                    double[] a = new double[lpcOrder + 1];
                    double e = r[0];
                    a[0] = 1;
                    for (int i = 1; i <= lpcOrder; i++) {
                        double sum = 0;
                        for (int j = 1; j < i; j++) {
                            sum += a[j] * r[i - j];
                        }
                        k[i] = (r[i] - sum) / e;
                        a[i] = k[i];
                        for (int j = 1; j < i; j++) {
                            a[j] = a[j] - k[i] * a[i - j];
                        }
                        e *= (1 - k[i] * k[i]);
                    }
                    for (int i = 0;i<lpcOrder;i++){
                        lpcCoeffs[i] = k[i+1]*100;
                    }

                    Log.d("LPC features", Arrays.toString(lpcCoeffs));
                    lpcFeaturesList.add(lpcCoeffs);

                    return true;
                }

                @Override
                public void processingFinished() {
                }
            });

            // Add MFCC and Spectral features to the pipeline
            MFCC mfcc = new MFCC(windowSize, sampleRate, 13, 40, 300, 3000);
            dispatcher.addAudioProcessor(mfcc);

            // Create the feature vectors
            dispatcher.addAudioProcessor(new AudioProcessor() {
                @Override
                public boolean process(AudioEvent audioEvent) {
                   float[] k = mfcc.getMFCC();
                   double mfccFeatures[] = new double[k.length];
                   for(int i =0 ;i<k.length;i++){
                       mfccFeatures[i] = Double.parseDouble(String.valueOf(k[i]));
                   }
                   mfccFeaturesList.add(mfccFeatures);
                    Log.d("MFCC features", Arrays.toString(mfccFeatures));

                    // Create feature vector combining MFCC, LPC, and Spectral features
                    double[] featureVector = new double[mfccFeatures.length + lpcOrder + 2];

                    double f = calculateSpectralCentroid(audioEvent);
                    double g = calculateSpectralRollOff(audioEvent);
                    Log.d("Special features", Double.toString(f)+" @nd fetures-"+Double.toString(g));
                    featureVector[featureVector.length - 2] = f;
                    featureVector[featureVector.length - 1] = g;
                    double[] h = {f,g};
                    spectralFeaturesList.add(h);

                    return true;
                }

                @Override
                public void processingFinished() {
                }
            });

            // Run the pipeline
            dispatcher.run();

        } catch ( IOException e) {
            e.printStackTrace();
        }

    }

    private static double calculateSpectralCentroid(AudioEvent audioEvent) {
        float[] spectrum = audioEvent.getFloatBuffer();
        double weightedSum = 0.0;
        double sum = 0.0;
        for (int i = 0; i < spectrum.length; i++) {
            weightedSum += i * spectrum[i];
            sum += spectrum[i];
        }
        return weightedSum / sum;
    }

    private static double calculateSpectralRollOff(AudioEvent audioEvent) {
        float[] spectrum = audioEvent.getFloatBuffer();
        double totalEnergy = 0.0;
        for (double value : spectrum) {
            totalEnergy += value;
        }
        double threshold = 0.85 * totalEnergy;
        double cumulativeEnergy = 0.0;
        int bin = 0;
        while (cumulativeEnergy < threshold && bin < spectrum.length) {
            cumulativeEnergy += spectrum[bin];
            bin++;
        }
        return bin;
    }
    public static List<double[]> extractMFCC() {
        return mfccFeaturesList;
    }

    public static List<double[]> extractLPC() {
        return lpcFeaturesList;
    }
    public static List<double[]> extractSpectralFeatures() {
        return spectralFeaturesList;
    }
}
