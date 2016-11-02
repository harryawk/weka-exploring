/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucil.dua.ai;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Toshiba
 */
public class TucilDuaAi {
    
    static Instances datas;
    
    public static void LoadData() throws Exception {
        datas = DataSource.read
        ("C:\\Program Files\\Weka-3-8\\data\\iris.arff");
        datas.setClassIndex(datas.numAttributes()-1); //Set label atribute
    }
    
    public static Instances Discretize() throws Exception {
        Discretize discretize = new Discretize();
        String[] options = new String[2];
        options[0] = "-R";                                    
        options[1] = "1";                                     
        discretize.setOptions(options);
        discretize.setInputFormat(datas);
        return Filter.useFilter(datas,discretize);
    }
    
    public static void fullTrainingSet() throws Exception {
        Classifier j48 = new J48();
        j48.buildClassifier(datas);
        
        Evaluation eval = new Evaluation(datas);
        eval.evaluateModel(j48, datas);
        System.out.println("=====Run Information======");
        System.out.println("======Classifier Model======");
        System.out.println(j48.toString());
        System.out.println(eval.toSummaryString("====Stats======\n", false));
        System.out.println(eval.toClassDetailsString("====Detailed Result=====\n"));
        System.out.println(eval.toMatrixString("======Confusion Matrix======\n"));    
    }
    
    public static void crossValidation() throws Exception {
        Evaluation evaluation = new Evaluation(datas);
        J48 attr_tree = new J48();
        attr_tree.buildClassifier(datas);
        evaluation.crossValidateModel(attr_tree, datas, 10, new Random(1));
        System.out.println("=====Run Information======");
        System.out.println("======Classifier Model======");
        System.out.println(attr_tree.toString());
        System.out.println(evaluation.toSummaryString("====Stats======\n", false));
        System.out.println(evaluation.toClassDetailsString("====Detailed Result=====\n"));
        System.out.println(evaluation.toMatrixString("======Confusion Matrix======\n"));
    }
    
    public static void main(String[] args) {
        // TODO code application logic here
        TucilDuaAi test= new TucilDuaAi();
        try {
            LoadData();
            fullTrainingSet();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
}
