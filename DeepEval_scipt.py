from deepeval import evaluate
from deepeval.metrics import HallucinationMetric, BiasMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import pandas as pd
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import (
    HallucinationMetric, 
    BiasMetric, 
    AnswerRelevancyMetric,
    GEval,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

# Define the actual captions and generated captions
actual_captions = [
    "Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels over a prolonged period of time. The symptoms of diabetes mellitus can vary, depending on the type and severity of the condition. However, some common symptoms include frequent urination, excessive thirst, unexplained weight loss, increased hunger, fatigue, blurred vision, slow healing of wounds, and recurrent infections. It's important to note that these symptoms can also be present in other medical conditions, so it's essential to consult with a healthcare professional for an accurate diagnosis.",
    #"Hypertension is a common problem in the elderly population and is associated with increased cardiovascular risk. The management of hypertension in the elderly should be individualized and take into account the patient's overall medical condition, comorbidities, and treatment goals. 1. Lifestyle modifications: Encourage patients to adopt a healthy lifestyle, including regular exercise, weight loss if overweight, moderation of alcohol intake, and a low-sodium diet. 2. Medications: Initiate antihypertensive medication therapy in patients with stage 1 hypertension (systolic BP 140-159 mmHg or diastolic BP 90-99 mmHg) who have cardiovascular disease, target organ damage, diabetes, or a 10-year atherosclerotic cardiovascular disease (ASCVD) risk of 10% or higher. For patients without these conditions, consider starting treatment if the blood pressure is consistently 150/90 mmHg or higher. 3. Choice of antihypertensive medications: The choice of medication should be tailored to the individual patient, taking into account their comorbidities, potential side effects, and drug interactions. Commonly used classes of antihypertensive medications in the elderly include angiotensin-converting enzyme inhibitors (ACE inhibitors), angiotensin receptor blockers (ARBs), calcium channel blockers (CCBs), diuretics, beta-blockers, and alpha-blockers. 4. Blood pressure targets: The target blood pressure for most elderly patients with hypertension is typically a systolic BP less than 150 mmHg and a diastolic BP less than 90 mmHg. However, individualized approaches are necessary, considering factors such as age, comorbidities, and treatment goals. 5. Regular monitoring: Regular follow-up visits are essential to assess blood pressure control, adjust medications as needed, and monitor for any adverse effects. 6. Other considerations: Assess and manage any underlying conditions that contribute to hypertension, such as sleep apnea or Cushing's syndrome. Be aware of the risks and benefits of specific antihypertensive medications in elderly patients, taking into account their renal",
    "Statins are medications commonly prescribed to lower cholesterol levels in the blood. They work by blocking an enzyme that is involved in the production of cholesterol in the liver. While statins are generally safe and effective, they can cause some side effects. The most common side effects of statins include: 1. Muscle pain and weakness: Statins can cause muscle pain, tenderness, or weakness. This is because statins can break down muscle tissue, leading to inflammation and discomfort. 2. Liver damage: In rare cases, statins can cause liver damage. Regular monitoring of liver function is recommended for individuals taking statins. 3. Digestive problems: Some people may experience gastrointestinal issues such as nausea, diarrhea, or constipation while taking statins. 4. Increased blood sugar levels: Statins have been associated with a slight increase in blood sugar levels. This can be problematic for individuals who already have diabetes or are at risk of developing it. 5. Memory loss or confusion: Although rare, some individuals may experience memory loss or confusion while taking statins. 6. Rhabdomyolysis: This is a rare but serious side effect of statins. It involves the breakdown of muscle tissue, leading to the release of muscle fiber contents into the bloodstream. This can cause kidney damage and other complications. It's important to note that most people tolerate statins well and experience few or no side effects. However, if you experience any concerning symptoms while taking statins, it's important to consult with your healthcare provider. They can help determine if the benefits of the medication outweigh the potential risks or adjust the dosage if necessary.",
    "Pneumonia is diagnosed through a combination of clinical signs and laboratory tests. Clinical signs of pneumonia may include: - Cough - often with phlegm that may be yellow or green - Fever - Chest pain when breathing or coughing - Rapid heartbeat - Shortness of breath - Fatigue or weakness - Sweating - Chills - Headache - Confusion (in older adults) If a doctor suspects pneumonia based on the patient’s symptoms, he or she will likely order laboratory tests to confirm the diagnosis. These tests may include: - Chest X-ray - this can show areas of infection in the lungs. - Blood tests - to check for signs of infection, such as increased white blood cells. - Sputum culture - a sample of the mucus coughed up from the lungs is tested to identify the type of infection causing pneumonia. - Blood cultures - in cases of severe pneumonia, a blood sample may be taken to check for bacteria or fungi in the bloodstream. Once a diagnosis of pneumonia is made, the doctor will determine the severity of the disease and whether it is community-acquired or hospital-acquired. This information helps guide treatment decisions.",
    "There are several treatment options for migraine headaches.  The first step is to make lifestyle changes that can help reduce the frequency and severity of the headaches.  This includes getting regular exercise, maintaining a consistent sleep schedule, managing stress, and avoiding triggers such as certain foods and drinks. For acute treatment of a migraine attack, over-the-counter pain relievers such as ibuprofen or aspirin may be effective for mild to moderate migraines.  However, for more severe migraines, prescription medications may be needed.  These can include triptans, which help to relieve pain and reduce inflammation, or ergotamine derivatives, which constrict blood vessels in the brain. In some cases, preventive medications may be prescribed to help reduce the frequency and severity of migraines.  These medications are usually taken on a daily basis and can include beta blockers, anticonvulsants, antidepressants, or Botox injections. It is important to work with a healthcare provider to determine the most appropriate treatment plan for your specific migraines.",
    "Coronary artery disease (CAD) is a condition in which a waxy substance called plaque builds up inside of the coronary arteries. These arteries supply oxygen-rich blood to your heart. When plaque builds up in the arteries, it narrows them, reducing blood flow to your heart. This can lead to chest pain (angina), heart attack, and other complications. There are several risk factors for CAD. Some of the risk factors you cannot change, such as your age, sex, and family history. However, there are many risk factors that you can change or manage. These include: 1. High blood pressure: High blood pressure can damage your artery walls, allowing plaque to build up more quickly. 2. High cholesterol levels: High levels of cholesterol in your blood can lead to the formation of plaques in your arteries. 3. Diabetes: Having diabetes doubles your risk of developing CAD. 4. Smoking: Smoking damages your blood vessels and speeds up the process of plaque buildup. 5. Obesity: Being overweight or obese increases your risk of CAD and other heart diseases. 6. Physical inactivity: Lack of exercise can lead to weight gain, high blood pressure, and high cholesterol, which are all risk factors for CAD. 7. Unhealthy diet: A diet high in saturated fat, trans fat, and cholesterol can increase your risk of CAD. 8. Stress: Chronic stress may contribute to the development of CAD. It is important to know your risk factors for CAD. If you have risk factors that you cannot change, such as your age or family history, it is even more important to manage the risk factors that you can. Making lifestyle changes, such as eating a healthy diet, exercising regularly, not smoking, and managing other health conditions like diabetes or high blood pressure, can help reduce your risk of developing CAD."
]

generated_captions = [
    "Diabetes mellitus is a metabolic disorder characterized by high blood glucose levels. The most common signs and symptoms of diabetes include frequent urination (polyuria), excessive thirst (polydipsia), unexplained weight loss, increased hunger, fatigue, blurred vision, slow-healing sores or frequent infections, and darkened patches of skin, especially around the neck and armpits (acanthosis nigricans). If you experience any of these symptoms, it is important to consult with a healthcare professional for proper evaluation and diagnosis."
    #"Hypertension, or high blood pressure, is a common condition that can increase the risk of heart disease, stroke, and other health problems. Managing hypertension in older adults involves a combination of lifestyle changes and medication. Here are some recommended strategies for managing hypertension in older adults: 1. Monitor blood pressure regularly: Regular monitoring of blood pressure is essential to ensure it stays within a healthy range. This can be done at home using a reliable blood pressure monitor or through regular visits to the doctor. 2. Adopt a healthy diet: Following a balanced diet that is low in sodium (salt), saturated fats, and cholesterol can help lower blood pressure. Emphasize fruits, vegetables, whole grains, lean proteins, and low-fat dairy products. 3. Maintain a healthy weight: Losing excess weight can have a significant impact on blood pressure. Aim for a body mass index (BMI) within the normal range (18.5-24.9). 4. Engage in regular physical activity: Regular exercise can help lower blood pressure. Aim for at least 150 minutes of moderate-intensity aerobic activity per week, such as brisk walking or cycling. 5. Limit alcohol consumption: Excessive alcohol intake can raise blood pressure. Older adults should limit alcohol consumption to no more than one drink per day for women and two drinks per day for men. 6. Quit smoking: Smoking can elevate blood pressure and damage blood vessels. Quitting smoking is important for overall cardiovascular health. 7. Manage stress: Chronic stress can contribute to high blood pressure. Engaging in relaxation techniques like deep breathing exercises, meditation, or yoga can help manage stress levels. 8. Take prescribed medications: In addition to lifestyle modifications, older adults may require medication to manage hypertension. It is important to take prescribed medications as directed by a healthcare professional. 9. Follow up with healthcare providers: Regular follow-up appointments with healthcare providers are crucial to monitor blood pressure levels and adjust treatment plans as needed. In summary, managing hypertension in older adults involves a combination of lifestyle changes (such as adopting a healthy diet, maintaining a healthy weight, engaging in regular physical activity, limiting alcohol consumption, quitting smoking, managing stress",
    "Statins are commonly used to lower cholesterol levels in patients. While they are generally safe and well tolerated, they can cause some adverse effects. These include: 1. Muscle pain and weakness: Statins can cause muscle inflammation (myopathy) and muscle damage (rhabdomyolysis), which can lead to pain, weakness and in severe cases, kidney failure. 2. Liver damage: Statins can rarely cause liver enzyme abnormalities, although serious liver problems are very rare. 3. Digestive problems: Some people may experience nausea, vomiting, diarrhea or constipation. 4. Increased blood sugar levels: Statins can slightly increase blood sugar levels, which may be a concern for people with diabetes or prediabetes. 5. Neurological effects: There have been reports of memory loss, confusion and cognitive impairment associated with statin use, although the evidence is inconclusive. It is important to note that the benefits of statins in reducing the risk of cardiovascular events generally outweigh the potential risks of these adverse effects. However, if you experience any concerning side effects while taking statins, it is important to discuss them with your doctor. The Answer is: There can be several adverse effects from taking statins, including muscle pain and weakness, liver damage, digestive problems, increased blood sugar levels, and neurological effects."
    "- A chest X-ray is often used to diagnose pneumonia. - A sputum culture may be ordered to identify the specific germ causing the infection. - Blood tests can show an increased white blood cell count. - A pulse oximetry test measures the amount of oxygen in the blood. - In severe cases, a bronchoscopy may be performed to obtain a sample of the infected lung tissue. How is pneumonia treated? - Antibiotics are used to treat bacterial pneumonia. - Antiviral medications may be prescribed for viral pneumonia. - Supportive care, such as rest, fluids, and over-the-counter pain relievers, can help alleviate symptoms. - In severe cases, hospitalization may be necessary. - People with weakened immune systems may require stronger antibiotics or other treatments. What are the complications of pneumonia? - Pneumonia can lead to respiratory failure if the infection spreads to the air sacs in the lungs. - It can also cause sepsis, a life-threatening condition that occurs when the infection spreads throughout the body. - Other complications may include fluid in the lungs, lung abscesses, and worsening of underlying health conditions. How can pneumonia be prevented? - Vaccination against pneumococcal pneumonia is recommended for adults 65 years and older, as well as for those with certain medical conditions. - Practicing good hand hygiene by washing hands regularly can help prevent the spread of germs. - Avoiding close contact with individuals who have pneumonia can reduce the risk of infection. - Prompt treatment of other respiratory infections, such as the flu, can help prevent secondary bacterial pneumonia.",
    "Migraine headaches are often treated with a combination of lifestyle changes, medications, and alternative therapies. Lifestyle changes that can help manage migraines include: - Identifying and avoiding triggers such as certain foods, stress, lack of sleep, and bright lights. - Maintaining a regular sleep schedule and getting enough rest. - Managing stress through relaxation techniques, meditation, or counseling. - Regular exercise to help reduce the frequency and severity of migraines. Medications commonly prescribed for migraines include: - Over-the-counter pain relievers like acetaminophen, ibuprofen, or aspirin. - Prescription pain medications such as triptans or ergotamine derivatives. - Anti-nausea medications to alleviate symptoms during a migraine attack. - Preventive medications, such as beta-blockers or antidepressants, to reduce the frequency and severity of migraines. Alternative therapies that may be helpful include: - Acupuncture: This traditional Chinese medicine technique involves inserting thin needles into specific points on the body to promote relaxation and relieve pain. - Biofeedback: This technique helps individuals learn how to control physical responses associated with migraines, such as heart rate and muscle tension. - Mindfulness meditation: This practice involves focusing attention on the present moment to reduce stress and promote relaxation. It is important to work closely with a healthcare provider to develop an individualized treatment plan based on the severity and frequency of migraines and personal preferences.",
    "Coronary artery disease (CAD) is a condition where plaque builds up inside the coronary arteries, which supply blood and oxygen to the heart muscle. Several factors increase the likelihood of developing CAD: 1. Age: The risk of CAD increases with age. Men over 45 years and women over 55 years have a higher risk. 2. Family history: Having a close family member, such as a parent or sibling, with a history of CAD increases the risk. 3. Gender: Men are generally at a higher risk of developing CAD than premenopausal women. However, after menopause, the risk for women increases and becomes similar to that of men. 4. Smoking: Tobacco smoke contains chemicals that can damage the inner lining of the arteries, leading to plaque formation. Smokers are at a higher risk of developing CAD. 5. High blood pressure: High blood pressure puts extra strain on the heart and arteries, increasing the risk of CAD. 6. High cholesterol levels: Elevated levels of LDL (bad) cholesterol and triglycerides, along with low levels of HDL (good) cholesterol, contribute to the buildup of plaque in the arteries. 7. Diabetes: People with diabetes have an increased risk of developing CAD. 8. Obesity: Being overweight or obese increases the risk of CAD. 9. Sedentary lifestyle: Lack of regular physical activity is associated with a higher risk of CAD. 10. Stress: Chronic stress may contribute to the development of CAD by causing changes in blood pressure, heart rate, and blood clotting. It's important to note that having these risk factors doesn't necessarily mean someone will develop CAD. However, being aware of these factors can help in adopting lifestyle modifications to reduce the risk. The Answer is: These factors include age, family history, gender, smoking, high blood pressure, high cholesterol levels, diabetes, obesity, sedentary lifestyle, and stress.",
    ]
# Updated input prompt for a generic medical SLM
input_prompt = "As a medical AI assistant, analyze the following medical information and provide a detailed interpretation. Include relevant medical concepts, potential diagnoses, and clinical significance. Base your analysis solely on the information provided."

# Create test cases with proper parameters for all metrics
test_cases = []
for i in range(len(actual_captions)):
    test_case = LLMTestCase(
        input=input_prompt,
        actual_output=generated_captions[i],
        expected_output=actual_captions[i],  # Using actual caption as expected output for GEval
        context=[actual_captions[i]],
        retrieval_context=[actual_captions[i]]  # Adding retrieval_context for Faithfulness metric
    )
    test_cases.append(test_case)

# Initialize metrics with appropriate thresholds
hallucination_metric = HallucinationMetric(threshold=0.8)
bias_metric = BiasMetric(threshold=0.8)
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
faithfulness_metric = FaithfulnessMetric(threshold=0.7)
contextual_precision_metric = ContextualPrecisionMetric(threshold=0.7)
contextual_recall_metric = ContextualRecallMetric(threshold=0.7)
contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)

# Initialize GEval metric
geval_metric = GEval(
    name="Medical Accuracy",
    criteria="Evaluate if the model's response is medically accurate and complete compared to the reference answer.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "Check if the model's response contains any medical inaccuracies or factual errors",
        "Assess whether the model's response covers the key medical concepts mentioned in the reference",
        "Determine if the model uses appropriate medical terminology",
        "Verify that the model doesn't make inappropriate claims about diagnoses or treatments",
        "Consider whether the model appropriately acknowledges limitations in medical knowledge when relevant"
    ],
    threshold=0.7
)

# Function to evaluate a single test case with all metrics
def evaluate_test_case(test_case, index):
    results_dict = {
        "Image Index": index + 1,
        "Actual Caption": test_case.context[0] if test_case.context else "",
        "Generated Caption": test_case.actual_output
    }
    
    # List of metrics to evaluate
    metrics = [
        ("Hallucination", hallucination_metric),
        ("Bias", bias_metric),
        ("Answer Relevancy", answer_relevancy_metric),
        ("Faithfulness", faithfulness_metric),
        ("Contextual Precision", contextual_precision_metric),
        ("Contextual Recall", contextual_recall_metric),
        ("Contextual Relevancy", contextual_relevancy_metric),
        ("GEval", geval_metric)
    ]
    
    # Evaluate each metric with error handling
    for metric_name, metric in metrics:
        try:
            metric.measure(test_case)
            results_dict[f"{metric_name} Score"] = metric.score
            results_dict[f"{metric_name} Reason"] = metric.reason
            results_dict[f"{metric_name} Pass"] = metric.score <= metric.threshold
        except Exception as e:
            print(f"Error measuring {metric_name}: {str(e)}")
            results_dict[f"{metric_name} Score"] = None
            results_dict[f"{metric_name} Reason"] = f"Error: {str(e)}"
            results_dict[f"{metric_name} Pass"] = False
    
    return results_dict

# Evaluate all test cases
results = []
for i, test_case in enumerate(test_cases):
    print(f"Evaluating case {i+1}/{len(test_cases)}...")
    result = evaluate_test_case(test_case, i)
    results.append(result)
    
    # Print scores for each metric
    for metric_name in ["Hallucination", "Bias", "Answer Relevancy", "Faithfulness", 
                        "Contextual Precision", "Contextual Recall", "Contextual Relevancy", "GEval"]:
        score = result.get(f"{metric_name} Score")
        passed = result.get(f"{metric_name} Pass")
        if score is not None:
            print(f"  {metric_name} Score: {score:.2f} ({'PASS' if passed else 'FAIL'})")
        else:
            print(f"  {metric_name}: Error occurred")

# Create a DataFrame for better visualization
df_results = pd.DataFrame(results)

# Print summary statistics
print("\n=== EVALUATION SUMMARY ===")
print(f"Total Cases: {len(test_cases)}")

# Calculate passes for each metric
for metric_name in ["Hallucination", "Bias", "Answer Relevancy", "Faithfulness", 
                    "Contextual Precision", "Contextual Recall", "Contextual Relevancy", "GEval"]:
    pass_col = f"{metric_name} Pass"
    if pass_col in df_results.columns:
        passes = sum(df_results[pass_col].fillna(False))
        print(f"{metric_name} Passes: {passes}/{len(test_cases)}")

# Calculate overall passes (all metrics)
metric_pass_columns = [col for col in df_results.columns if col.endswith(" Pass")]
if metric_pass_columns:
    overall_passes = df_results[metric_pass_columns].all(axis=1).sum()
    print(f"Overall Passes (all metrics): {overall_passes}/{len(test_cases)}")

# Save results to CSV
df_results.to_csv("medical_slm_evaluation.csv", index=False)
print("\nDetailed results saved to 'medical_slm_evaluation.csv'")

# Function to run a full evaluation using DeepEval's evaluate function
def run_full_evaluation():
    print("\nRunning full evaluation with DeepEval...")
    all_metrics = [
        hallucination_metric, 
        bias_metric, 
        answer_relevancy_metric,
        faithfulness_metric,
        contextual_precision_metric,
        contextual_recall_metric,
        contextual_relevancy_metric,
        geval_metric
    ]
    try:
        evaluate(test_cases, all_metrics)
    except Exception as e:
        print(f"Error in full evaluation: {str(e)}")

# Run the full evaluationc
run_full_evaluation()