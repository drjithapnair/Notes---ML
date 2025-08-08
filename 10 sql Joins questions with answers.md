
# 10 sql Joins questions with answers  also create 2 tables and insert 10 values each table name student and mark

Sure! Below are two sample tables `student` and `mark` with 10 rows each. After that, I will provide 10 SQL JOIN questions along with their answers based on these tables.

***

### Step 1: Create Tables and Insert Data

```sql
CREATE TABLE student (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(50),
    age INT,
    city VARCHAR(50)
);

CREATE TABLE mark (
    mark_id INT PRIMARY KEY,
    student_id INT,
    subject VARCHAR(50),
    score INT,
    FOREIGN KEY (student_id) REFERENCES student(student_id)
);

-- Insert data into student table
INSERT INTO student (student_id, student_name, age, city) VALUES
(1, 'Alice', 20, 'New York'),
(2, 'Bob', 22, 'Los Angeles'),
(3, 'Charlie', 21, 'Chicago'),
(4, 'David', 23, 'Houston'),
(5, 'Eve', 20, 'Phoenix'),
(6, 'Frank', 22, 'Philadelphia'),
(7, 'Grace', 21, 'San Antonio'),
(8, 'Hannah', 23, 'San Diego'),
(9, 'Ian', 20, 'Dallas'),
(10, 'Jane', 22, 'San Jose');

-- Insert data into mark table
INSERT INTO mark (mark_id, student_id, subject, score) VALUES
(1, 1, 'Math', 85),
(2, 1, 'Science', 90),
(3, 2, 'Math', 78),
(4, 3, 'History', 88),
(5, 4, 'Math', 92),
(6, 5, 'Science', 75),
(7, 6, 'History', 80),
(8, 7, 'Math', 70),
(9, 8, 'Science', 95),
(10, 10, 'History', 85);
```


***

### Step 2: 10 SQL JOIN Questions with Answers


***

#### 1. **Inner Join: Get students with their marks for all subjects.**

```sql
SELECT s.student_id, s.student_name, m.subject, m.score
FROM student s
INNER JOIN mark m ON s.student_id = m.student_id;
```


***

#### 2. **Left Join: Get all students and their marks, include students with no marks.**

```sql
SELECT s.student_id, s.student_name, m.subject, m.score
FROM student s
LEFT JOIN mark m ON s.student_id = m.student_id;
```


***

#### 3. **Right Join: Get all marks and student details, including marks with no student (if any).**

```sql
SELECT s.student_id, s.student_name, m.subject, m.score
FROM student s
RIGHT JOIN mark m ON s.student_id = m.student_id;
```


***

#### 4. **Full Outer Join: Get all students and marks, with matching or non-matching records.**

Note: Full outer join syntax may differ based on SQL flavor (not supported natively in MySQL).

```sql
-- Use UNION of LEFT and RIGHT JOIN to simulate FULL OUTER JOIN in MySQL
SELECT s.student_id, s.student_name, m.subject, m.score
FROM student s
LEFT JOIN mark m ON s.student_id = m.student_id
UNION
SELECT s.student_id, s.student_name, m.subject, m.score
FROM student s
RIGHT JOIN mark m ON s.student_id = m.student_id;
```


***

#### 5. **Get students who have marks recorded (use Inner Join).**

```sql
SELECT s.student_id, s.student_name
FROM student s
INNER JOIN mark m ON s.student_id = m.student_id
GROUP BY s.student_id, s.student_name;
```


***

#### 6. **Get students who do NOT have any marks recorded (use Left Join and filter NULL).**

```sql
SELECT s.student_id, s.student_name
FROM student s
LEFT JOIN mark m ON s.student_id = m.student_id
WHERE m.mark_id IS NULL;
```


***

#### 7. **Get total marks for each student (using Inner Join and GROUP BY).**

```sql
SELECT s.student_id, s.student_name, SUM(m.score) AS total_score
FROM student s
INNER JOIN mark m ON s.student_id = m.student_id
GROUP BY s.student_id, s.student_name;
```


***

#### 8. **Get the highest score of each student (using Inner Join and GROUP BY).**

```sql
SELECT s.student_id, s.student_name, MAX(m.score) AS highest_score
FROM student s
INNER JOIN mark m ON s.student_id = m.student_id
GROUP BY s.student_id, s.student_name;
```


***

#### 9. **Get students with their Math subject score only (use Inner Join and WHERE).**

```sql
SELECT s.student_id, s.student_name, m.score
FROM student s
INNER JOIN mark m ON s.student_id = m.student_id
WHERE m.subject = 'Math';
```


***

#### 10. **Get students along with the number of subjects they have marks for.**

```sql
SELECT s.student_id, s.student_name, COUNT(m.subject) AS subject_count
FROM student s
LEFT JOIN mark m ON s.student_id = m.student_id
GROUP BY s.student_id, s.student_name;
```


***


