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
import weka.core.SerializationHelper;

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
        attr_tree.buildClassifier(datas);
        evaluation.crossValidateModel(attr_tree, datas, 10, new Random(1));
        System.out.println("=====Run Information======");
        System.out.println("======Classifier Model======");
        System.out.println(attr_tree.toString());
        System.out.println(evaluation.toSummaryString("====Stats======\n", true));
        System.out.println(evaluation.toClassDetailsString("====Detailed Result=====\n"));
        System.out.println(evaluation.toMatrixString("======Confusion Matrix======\n"));
//        System.out.println(evaluation.("======Confusion Matrix======\n"));
        try {
            SaveModel(attr_tree);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("===========================");
        System.out.println("===========================");
        System.out.println("===========================");
        System.out.println("===========================");
        System.out.println("===========================");
        try {
            Evaluation ev = new Evaluation(datas);
            J48 attr_ = LoadModel();
            attr_.buildClassifier(datas);
            ev.crossValidateModel(attr_, datas, 10, new Random(1));
            System.out.println("=====Run Information======");
            System.out.println("======Classifier Model======");
            System.out.println(attr_.toString());
            System.out.println(ev.toSummaryString("====Stats======\n", true));
            System.out.println(ev.toClassDetailsString("====Detailed Result=====\n"));
            System.out.println(ev.toMatrixString("======Confusion Matrix======\n"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    /**
     * 
     * @throws Exception 
     */
    public static void LoadData() throws Exception {
        DataSource source = new DataSource("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
        datas = source.getDataSet();
        datas.setClassIndex(datas.numAttributes()-1); // Set label atribut
//        System.out.println(datas);
    }
    public static void SaveModel(J48 ins) throws Exception {
//        Sebelumnya harus terdapat direktori C:\goog\
        SerializationHelper.write("C:\\goog\\j48.model", ins);
        System.out.println("Uploaded");
    }
    public static J48 LoadModel() throws Exception {
        J48 temp = (J48) SerializationHelper.read("C:\\goog\\j48.model");
        return temp;
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
////        System.out.println(datas.get(1).dataset());
//        double[] dabel = datas.instance(0).toDoubleArray();
////        System.out.println(datas.instance(0).toDoubleArray());
//        Instance data;
//        for (int i = 0; i < dabel.length; i++) {
//            System.out.println(dabel[i]);
//        }
    }    
}
