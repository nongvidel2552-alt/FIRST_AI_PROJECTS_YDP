# 🏦 Bank Customer Churn Prediction

## 🎯 Project Overview
โปรเจกต์นี้มีเป้าหมายเพื่อ ทำนายว่าลูกค้าธนาคารมีแนวโน้มจะ “ออก” หรือ “อยู่”
โดยใช้โมเดล Machine Learning เพื่อช่วยให้ธนาคาร **ลดการสูญเสียลูกค้า** และ **รักษาลูกค้าสำคัญไว้ได้อย่างมีประสิทธิภาพ** 

---

## 📊 Dataset
- ข้อมูลลูกค้าธนาคาร (เช่น อายุ ประเทศ เงินเดือน บัญชี/ผลิตภัณฑ์ที่ใช้)
- เป้าหมาย (Target): `Exited` — 0 = อยู่, 1 = ออก

---
Tech stacks:
pandas
numpy
matplotlib
seaborn
scikit-learn

## 🔍 Workflow
1. **EDA (Exploratory Data Analysis)**  
   - วิเคราะห์ข้อมูลเบื้องต้น  
   - ดูการกระจายของค่า และ Exit Rate  

2. **Feature Engineering (3 มิติ)**  
   - Concept-based (Demographic / Behavior / Financial)  
   - Relationship-based (Engagement / Loyalty / Financial Power)  
   - Numeric binning (Age / Balance / Salary / Points / Tenure)

3. **Data Preprocessing**  
   - Encoding (One-Hot)  
   - Scaling (StandardScaler)  
   - Train-Test Split (80/20, Stratify)

4. **Modeling**  
   - Logistic Regression → เพื่ออธิบายเหตุผลรายลูกค้า  
   - Random Forest → เพื่อเพิ่มความแม่นยำโดยรวม  

5. **Targeted Action Plan**  
   - วิเคราะห์ Top 1–2 เหตุผลที่ลูกค้ามีแนวโน้มจะออก  
   - แมปเหตุผล → ข้อเสนอเฉพาะบุคคล (Personalized Retention)

---

## 🧠 Model Performance
| Model                    | Accuracy | Precision | Recall | F1-score |
|---------------------------|-----------|-----------|--------|-----------|
| Logistic Regression       | 0.728     | 0.404     | 0.706  | 0.514     |
| Random Forest             | 0.865     | 0.758     | 0.493  | 0.597     |

---

## ✨ Highlights
- 🧠 คิด Feature Engineering  (3-Dimensional Feature Strategy)  
- 📈 ใช้ 2 โมเดล เพื่อ balance “ความแม่นยำ” และ “ความเข้าใจ”  
- 🧭 วิเคราะห์สาเหตุรายบุคคล + ข้อเสนอ Retention เฉพาะลูกค้า (Targeted Action, Not Generic Response)

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python your_script_name.py
