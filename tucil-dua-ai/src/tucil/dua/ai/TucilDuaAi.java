/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil.dua.ai;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.Instance;

/**
 *
 * @author Toshiba
 */
public class TucilDuaAi {
    
    static Instances datas;
    /**
     * 
     */
    public static void Classifier() throws Exception {
        Evaluation evaluation = new Evaluation(datas);
        J48 attr_tree = new J48();
        evaluation.crossValidateModel(attr_tree, datas, 10, new Random(1));
        System.out.println(evaluation.toSummaryString("\nResults\n\n", false));
    }
    /**
     * 
     * @throws Exception 
     */
    public static void LoadData() throws Exception {
        datas = DataSource.read("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
//        System.out.println(datas);
        datas.setClassIndex(datas.numAttributes()-1); // Set label atribut
    }
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        try {
            LoadData();
            Classifier();
        } catch (Exception e) {
            e.printStackTrace();
        }
//        System.out.println(datas.get(1).dataset());
        double[] dabel = datas.instance(0).toDoubleArray();
//        System.out.println(datas.instance(0).toDoubleArray());
        Instance data;
        for (int i = 0; i < dabel.length; i++) {
            System.out.println(dabel[i]);
        }
    }
    
}
