# Complete SQL Commands Reference

## Data Definition Language (DDL) Commands

### DATABASE Commands
```sql
-- Create Database
CREATE DATABASE database_name;
CREATE DATABASE IF NOT EXISTS company_db;

-- Drop Database
DROP DATABASE database_name;
DROP DATABASE IF EXISTS company_db;

-- Show Databases
SHOW DATABASES;

-- Use Database
USE database_name;

-- Show Current Database
SELECT DATABASE();
```

### TABLE Commands
```sql
-- Create Table
CREATE TABLE table_name (
    column1 datatype constraints,
    column2 datatype constraints,
    ...
);

CREATE TABLE students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INT CHECK (age >= 18),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Table from Another Table
CREATE TABLE new_table AS SELECT * FROM existing_table;
CREATE TABLE backup_students AS SELECT * FROM students WHERE age > 20;

-- Show Tables
SHOW TABLES;

-- Describe Table Structure
DESC table_name;
DESCRIBE students;
SHOW COLUMNS FROM students;

-- Show Create Table Statement
SHOW CREATE TABLE students;

-- Drop Table
DROP TABLE table_name;
DROP TABLE IF EXISTS students;

-- Truncate Table (Delete all data, keep structure)
TRUNCATE TABLE table_name;

-- Rename Table
RENAME TABLE old_name TO new_name;
ALTER TABLE students RENAME TO pupils;
```

### ALTER TABLE Commands
```sql
-- Add Column
ALTER TABLE students ADD COLUMN phone VARCHAR(15);
ALTER TABLE students ADD COLUMN (
    address TEXT,
    city VARCHAR(50)
);

-- Drop Column
ALTER TABLE students DROP COLUMN phone;
ALTER TABLE students DROP COLUMN address, DROP COLUMN city;

-- Modify Column
ALTER TABLE students MODIFY COLUMN name VARCHAR(150);
ALTER TABLE students MODIFY COLUMN age TINYINT;

-- Change Column (rename + modify)
ALTER TABLE students CHANGE COLUMN name full_name VARCHAR(150);

-- Add Constraint
ALTER TABLE students ADD CONSTRAINT unique_email UNIQUE (email);
ALTER TABLE students ADD CONSTRAINT check_age CHECK (age BETWEEN 18 AND 65);

-- Drop Constraint
ALTER TABLE students DROP CONSTRAINT unique_email;
ALTER TABLE students DROP INDEX unique_email;

-- Add Primary Key
ALTER TABLE students ADD PRIMARY KEY (id);

-- Drop Primary Key
ALTER TABLE students DROP PRIMARY KEY;

-- Add Foreign Key
ALTER TABLE enrollments ADD CONSTRAINT fk_student 
FOREIGN KEY (student_id) REFERENCES students(id);

-- Drop Foreign Key
ALTER TABLE enrollments DROP FOREIGN KEY fk_student;
```

### INDEX Commands
```sql
-- Create Index
CREATE INDEX idx_name ON students(name);
CREATE INDEX idx_name_age ON students(name, age);

-- Create Unique Index
CREATE UNIQUE INDEX idx_email ON students(email);

-- Show Indexes
SHOW INDEX FROM students;
SHOW INDEXES FROM students;

-- Drop Index
DROP INDEX idx_name ON students;
ALTER TABLE students DROP INDEX idx_name;
```

## Data Manipulation Language (DML) Commands

### INSERT Commands
```sql
-- Basic Insert
INSERT INTO students (name, email, age) 
VALUES ('John Doe', 'john@email.com', 22);

-- Multiple Inserts
INSERT INTO students (name, email, age) VALUES 
('Jane Smith', 'jane@email.com', 20),
('Bob Johnson', 'bob@email.com', 24),
('Alice Brown', 'alice@email.com', 19);

-- Insert All Columns
INSERT INTO students VALUES (1, 'Mike Wilson', 'mike@email.com', 23, NOW());

-- Insert from Another Table
INSERT INTO students (name, email, age)
SELECT name, email, age FROM temp_students;

-- Insert with Subquery
INSERT INTO high_performers (student_id, name)
SELECT id, name FROM students WHERE age > 21;

-- Insert Ignore (Skip duplicates)
INSERT IGNORE INTO students (name, email, age)
VALUES ('John Doe', 'john@email.com', 22);

-- Replace (Insert or Update if exists)
REPLACE INTO students (id, name, email, age)
VALUES (1, 'John Updated', 'john.new@email.com', 23);

-- On Duplicate Key Update
INSERT INTO students (name, email, age)
VALUES ('John Doe', 'john@email.com', 22)
ON DUPLICATE KEY UPDATE age = VALUES(age);
```

### UPDATE Commands
```sql
-- Basic Update
UPDATE students SET age = 25 WHERE id = 1;

-- Update Multiple Columns
UPDATE students 
SET name = 'John Smith', age = 26 
WHERE id = 1;

-- Update with Calculations
UPDATE students SET age = age + 1;

-- Update with Subquery
UPDATE students 
SET age = (SELECT AVG(age) FROM temp_students) 
WHERE id = 1;

-- Update Join
UPDATE students s
JOIN enrollments e ON s.id = e.student_id
SET s.name = CONCAT(s.name, ' (Enrolled)')
WHERE e.course_id = 101;

-- Conditional Update
UPDATE students 
SET name = CASE 
    WHEN age < 20 THEN CONCAT(name, ' (Young)')
    WHEN age > 25 THEN CONCAT(name, ' (Mature)')
    ELSE name
END;

-- Update with Limit
UPDATE students SET age = age + 1 ORDER BY id LIMIT 5;
```

### DELETE Commands
```sql
-- Basic Delete
DELETE FROM students WHERE id = 1;

-- Delete with Multiple Conditions
DELETE FROM students WHERE age < 18 OR email IS NULL;

-- Delete All Records
DELETE FROM students;

-- Delete with Subquery
DELETE FROM students 
WHERE id IN (SELECT student_id FROM temp_inactive);

-- Delete Join
DELETE s FROM students s
JOIN enrollments e ON s.id = e.student_id
WHERE e.course_id = 999;

-- Delete with Limit
DELETE FROM students ORDER BY created_at LIMIT 10;
```

## Data Query Language (DQL) Commands

### SELECT Commands
```sql
-- Basic Select
SELECT * FROM students;
SELECT name, email FROM students;

-- Select with Alias
SELECT name AS student_name, age AS student_age FROM students;

-- Select Distinct
SELECT DISTINCT age FROM students;

-- Select with Expressions
SELECT name, age, (age + 5) AS future_age FROM students;

-- Select Top/Limit
SELECT * FROM students LIMIT 5;
SELECT * FROM students LIMIT 5 OFFSET 10;

-- Select Random
SELECT * FROM students ORDER BY RAND() LIMIT 3;
```

### WHERE Conditions
```sql
-- Comparison Operators
SELECT * FROM students WHERE age = 20;
SELECT * FROM students WHERE age != 20;
SELECT * FROM students WHERE age <> 20;
SELECT * FROM students WHERE age > 20;
SELECT * FROM students WHERE age >= 20;
SELECT * FROM students WHERE age < 25;
SELECT * FROM students WHERE age <= 25;

-- Logical Operators
SELECT * FROM students WHERE age > 18 AND age < 25;
SELECT * FROM students WHERE name = 'John' OR name = 'Jane';
SELECT * FROM students WHERE NOT age = 20;

-- Range Operators
SELECT * FROM students WHERE age BETWEEN 18 AND 25;
SELECT * FROM students WHERE age NOT BETWEEN 18 AND 25;

-- Pattern Matching
SELECT * FROM students WHERE name LIKE 'J%';        -- Starts with J
SELECT * FROM students WHERE name LIKE '%son';      -- Ends with son
SELECT * FROM students WHERE name LIKE '%oh%';      -- Contains oh
SELECT * FROM students WHERE name LIKE 'J_hn';      -- J followed by any char, then hn
SELECT * FROM students WHERE name NOT LIKE 'A%';

-- Regular Expressions
SELECT * FROM students WHERE name REGEXP '^[A-M]';  -- Starts with A-M
SELECT * FROM students WHERE email REGEXP '.*@gmail\.com$';

-- NULL Checks
SELECT * FROM students WHERE email IS NULL;
SELECT * FROM students WHERE email IS NOT NULL;

-- IN Operator
SELECT * FROM students WHERE age IN (18, 20, 22);
SELECT * FROM students WHERE name IN ('John', 'Jane', 'Bob');
SELECT * FROM students WHERE id NOT IN (1, 2, 3);

-- EXISTS
SELECT * FROM students s 
WHERE EXISTS (SELECT 1 FROM enrollments e WHERE e.student_id = s.id);
```

### ORDER BY Commands
```sql
-- Single Column Sort
SELECT * FROM students ORDER BY age;
SELECT * FROM students ORDER BY age ASC;
SELECT * FROM students ORDER BY age DESC;

-- Multiple Column Sort
SELECT * FROM students ORDER BY age DESC, name ASC;

-- Sort by Expression
SELECT name, age, (age * 2) AS double_age 
FROM students 
ORDER BY double_age;

-- Sort by Column Position
SELECT name, age FROM students ORDER BY 2 DESC; -- Sort by age (2nd column)

-- Sort with NULL handling
SELECT * FROM students ORDER BY email IS NULL, email;
```

### GROUP BY and HAVING
```sql
-- Basic Group By
SELECT age, COUNT(*) FROM students GROUP BY age;

-- Group By with Multiple Columns
SELECT age, SUBSTRING(name, 1, 1) AS first_letter, COUNT(*)
FROM students 
GROUP BY age, first_letter;

-- Group By with Aggregate Functions
SELECT age, 
       COUNT(*) AS student_count,
       AVG(id) AS avg_id,
       MIN(name) AS first_name,
       MAX(name) AS last_name
FROM students 
GROUP BY age;

-- HAVING Clause
SELECT age, COUNT(*) as count
FROM students 
GROUP BY age
HAVING COUNT(*) > 2;

-- HAVING with Multiple Conditions
SELECT age, COUNT(*) as count, AVG(id) as avg_id
FROM students 
GROUP BY age
HAVING COUNT(*) > 1 AND AVG(id) > 5;

-- GROUP BY with ROLLUP
SELECT age, COUNT(*) FROM students GROUP BY age WITH ROLLUP;
```

## JOIN Commands

### INNER JOIN
```sql
-- Basic Inner Join
SELECT s.name, e.course_name
FROM students s
INNER JOIN enrollments e ON s.id = e.student_id;

-- Multiple Table Join
SELECT s.name, e.course_name, c.instructor
FROM students s
INNER JOIN enrollments e ON s.id = e.student_id
INNER JOIN courses c ON e.course_id = c.id;

-- Join with WHERE
SELECT s.name, e.course_name
FROM students s
INNER JOIN enrollments e ON s.id = e.student_id
WHERE s.age > 20;
```

### LEFT JOIN
```sql
-- Left Join (All students, even without enrollments)
SELECT s.name, e.course_name
FROM students s
LEFT JOIN enrollments e ON s.id = e.student_id;

-- Find students with no enrollments
SELECT s.name
FROM students s
LEFT JOIN enrollments e ON s.id = e.student_id
WHERE e.student_id IS NULL;
```

### RIGHT JOIN
```sql
-- Right Join (All enrollments, even if student doesn't exist)
SELECT s.name, e.course_name
FROM students s
RIGHT JOIN enrollments e ON s.id = e.student_id;
```

### FULL OUTER JOIN
```sql
-- Full Outer Join (MySQL doesn't support directly)
SELECT s.name, e.course_name
FROM students s
LEFT JOIN enrollments e ON s.id = e.student_id
UNION
SELECT s.name, e.course_name
FROM students s
RIGHT JOIN enrollments e ON s.id = e.student_id;
```

### SELF JOIN
```sql
-- Self Join (Find students with same age)
SELECT s1.name AS student1, s2.name AS student2, s1.age
FROM students s1
JOIN students s2 ON s1.age = s2.age AND s1.id < s2.id;
```

### CROSS JOIN
```sql
-- Cross Join (Cartesian Product)
SELECT s.name, c.course_name
FROM students s
CROSS JOIN courses c;
```

## Aggregate Functions

### Numeric Functions
```sql
-- COUNT
SELECT COUNT(*) FROM students;
SELECT COUNT(email) FROM students;  -- Excludes NULLs
SELECT COUNT(DISTINCT age) FROM students;

-- SUM
SELECT SUM(age) FROM students;
SELECT SUM(DISTINCT age) FROM students;

-- AVG
SELECT AVG(age) FROM students;
SELECT AVG(DISTINCT age) FROM students;

-- MIN/MAX
SELECT MIN(age), MAX(age) FROM students;
SELECT MIN(name), MAX(name) FROM students;  -- Alphabetical

-- STDDEV (Standard Deviation)
SELECT STDDEV(age) FROM students;

-- VARIANCE
SELECT VARIANCE(age) FROM students;
```

### String Functions
```sql
-- CONCAT
SELECT CONCAT(name, ' - Age: ', age) AS info FROM students;
SELECT CONCAT_WS(' | ', name, email, age) AS formatted FROM students;

-- LENGTH/CHAR_LENGTH
SELECT name, LENGTH(name) AS byte_length, CHAR_LENGTH(name) AS char_length 
FROM students;

-- SUBSTRING
SELECT SUBSTRING(name, 1, 3) AS first_three FROM students;
SELECT SUBSTRING(name FROM 2 FOR 3) AS middle_chars FROM students;

-- UPPER/LOWER
SELECT UPPER(name) AS upper_name, LOWER(email) AS lower_email FROM students;

-- TRIM
SELECT TRIM('  hello  ') AS trimmed;
SELECT LTRIM('  hello') AS left_trimmed;
SELECT RTRIM('hello  ') AS right_trimmed;

-- REPLACE
SELECT REPLACE(email, '@email.com', '@school.edu') AS school_email FROM students;

-- LEFT/RIGHT
SELECT LEFT(name, 3) AS first_three, RIGHT(name, 3) AS last_three FROM students;

-- LOCATE/POSITION
SELECT LOCATE('@', email) AS at_position FROM students;
SELECT POSITION('@' IN email) AS at_position FROM students;

-- REVERSE
SELECT REVERSE(name) FROM students;

-- REPEAT
SELECT REPEAT('*', 5) AS stars;

-- LPAD/RPAD
SELECT LPAD(name, 20, '-') AS padded_name FROM students;
SELECT RPAD('ID: ', 10, '0') AS padded_id;
```

### Date Functions
```sql
-- Current Date/Time
SELECT NOW(), CURDATE(), CURTIME();
SELECT CURRENT_TIMESTAMP, CURRENT_DATE, CURRENT_TIME;

-- Date Formatting
SELECT DATE_FORMAT(created_at, '%Y-%m-%d') AS date_only FROM students;
SELECT DATE_FORMAT(created_at, '%W, %M %d, %Y') AS readable_date FROM students;

-- Date Arithmetic
SELECT DATE_ADD(created_at, INTERVAL 1 YEAR) AS next_year FROM students;
SELECT DATE_SUB(NOW(), INTERVAL 30 DAY) AS thirty_days_ago;
SELECT DATEDIFF(NOW(), created_at) AS days_since_created FROM students;

-- Extract Date Parts
SELECT 
    YEAR(created_at) AS year,
    MONTH(created_at) AS month,
    DAY(created_at) AS day,
    HOUR(created_at) AS hour,
    MINUTE(created_at) AS minute,
    SECOND(created_at) AS second
FROM students;

-- Day of Week/Year
SELECT 
    DAYOFWEEK(created_at) AS day_of_week,
    DAYOFYEAR(created_at) AS day_of_year,
    WEEK(created_at) AS week_number
FROM students;

-- Last Day of Month
SELECT LAST_DAY(created_at) AS month_end FROM students;

-- Unix Timestamp
SELECT UNIX_TIMESTAMP(created_at) AS timestamp FROM students;
SELECT FROM_UNIXTIME(UNIX_TIMESTAMP()) AS from_timestamp;
```

### Mathematical Functions
```sql
-- Basic Math
SELECT ABS(-10) AS absolute;
SELECT CEIL(4.3) AS ceiling, FLOOR(4.7) AS floor;
SELECT ROUND(4.567, 2) AS rounded;
SELECT TRUNCATE(4.567, 1) AS truncated;

-- Power and Root
SELECT POWER(2, 3) AS power, SQRT(16) AS square_root;
SELECT EXP(1) AS exponential, LOG(10) AS natural_log;

-- Trigonometric
SELECT SIN(PI()/2) AS sine, COS(0) AS cosine, TAN(PI()/4) AS tangent;

-- Random
SELECT RAND() AS random_decimal;
SELECT FLOOR(RAND() * 100) AS random_integer;

-- Modulo
SELECT MOD(10, 3) AS remainder;

-- Sign
SELECT SIGN(-5) AS sign_negative, SIGN(5) AS sign_positive, SIGN(0) AS sign_zero;
```

## Control Flow Functions

### CASE Statements
```sql
-- Simple CASE
SELECT name, age,
    CASE age
        WHEN 18 THEN 'Just Adult'
        WHEN 19 THEN 'Teenager'
        WHEN 20 THEN 'Young Adult'
        ELSE 'Adult'
    END AS age_category
FROM students;

-- Searched CASE
SELECT name, age,
    CASE 
        WHEN age < 20 THEN 'Teenager'
        WHEN age BETWEEN 20 AND 25 THEN 'Young Adult'
        WHEN age > 25 THEN 'Adult'
        ELSE 'Unknown'
    END AS category
FROM students;
```

### IF Function
```sql
-- IF function
SELECT name, IF(age >= 21, 'Adult', 'Minor') AS status FROM students;

-- Nested IF
SELECT name, 
    IF(age < 20, 'Teen',
        IF(age < 30, 'Young Adult', 'Adult')
    ) AS category
FROM students;
```

### IFNULL and COALESCE
```sql
-- Handle NULL values
SELECT name, IFNULL(email, 'No Email') AS email_status FROM students;
SELECT name, COALESCE(email, phone, 'No Contact') AS contact FROM students;

-- NULLIF (return NULL if values are equal)
SELECT NULLIF(age, 0) AS valid_age FROM students;
```

## Window Functions (MySQL 8.0+)

### ROW_NUMBER, RANK, DENSE_RANK
```sql
-- Row Number
SELECT name, age, 
    ROW_NUMBER() OVER (ORDER BY age DESC) AS row_num
FROM students;

-- Rank functions
SELECT name, age,
    RANK() OVER (ORDER BY age DESC) AS rank_pos,
    DENSE_RANK() OVER (ORDER BY age DESC) AS dense_rank_pos
FROM students;

-- Partition by
SELECT name, age, 
    ROW_NUMBER() OVER (PARTITION BY age ORDER BY name) AS row_in_age_group
FROM students;
```

### LAG and LEAD
```sql
-- LAG (previous row)
SELECT name, age,
    LAG(age) OVER (ORDER BY id) AS previous_age,
    age - LAG(age) OVER (ORDER BY id) AS age_diff
FROM students;

-- LEAD (next row)
SELECT name, age,
    LEAD(age) OVER (ORDER BY id) AS next_age
FROM students;
```

### Aggregate Window Functions
```sql
-- Running totals
SELECT name, age,
    SUM(age) OVER (ORDER BY id ROWS UNBOUNDED PRECEDING) AS running_total
FROM students;

-- Moving average
SELECT name, age,
    AVG(age) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM students;
```

## Subquery Commands

### Scalar Subqueries
```sql
-- Single value subquery
SELECT name, age,
    (SELECT AVG(age) FROM students) AS avg_age
FROM students;

-- Correlated subquery
SELECT name, age
FROM students s1
WHERE age > (SELECT AVG(age) FROM students s2 WHERE s2.id != s1.id);
```

### Row Subqueries
```sql
-- Multiple column subquery
SELECT * FROM students
WHERE (age, name) = (SELECT MAX(age), MIN(name) FROM students);
```

### Table Subqueries
```sql
-- IN subquery
SELECT * FROM students
WHERE id IN (SELECT student_id FROM enrollments WHERE course_id = 101);

-- ANY/SOME subquery
SELECT * FROM students
WHERE age > ANY (SELECT age FROM students WHERE name LIKE 'J%');

-- ALL subquery
SELECT * FROM students
WHERE age > ALL (SELECT age FROM students WHERE name LIKE 'A%');

-- EXISTS subquery
SELECT * FROM students s
WHERE EXISTS (SELECT 1 FROM enrollments e WHERE e.student_id = s.id);

-- NOT EXISTS subquery
SELECT * FROM students s
WHERE NOT EXISTS (SELECT 1 FROM enrollments e WHERE e.student_id = s.id);
```

## Data Control Language (DCL) Commands

### User Management
```sql
-- Create User
CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';
CREATE USER 'username'@'%' IDENTIFIED BY 'password'; -- Any host

-- Drop User
DROP USER 'username'@'localhost';

-- Change Password
ALTER USER 'username'@'localhost' IDENTIFIED BY 'new_password';
SET PASSWORD FOR 'username'@'localhost' = PASSWORD('new_password');

-- Show Users
SELECT User, Host FROM mysql.user;
```

### Privilege Management
```sql
-- Grant Privileges
GRANT ALL PRIVILEGES ON database_name.* TO 'username'@'localhost';
GRANT SELECT, INSERT ON table_name TO 'username'@'localhost';
GRANT CREATE, DROP ON *.* TO 'username'@'localhost';

-- Revoke Privileges
REVOKE ALL PRIVILEGES ON database_name.* FROM 'username'@'localhost';
REVOKE SELECT ON table_name FROM 'username'@'localhost';

-- Show Privileges
SHOW GRANTS FOR 'username'@'localhost';
SHOW GRANTS FOR CURRENT_USER();

-- Flush Privileges
FLUSH PRIVILEGES;
```

## Transaction Control Language (TCL) Commands

### Transaction Commands
```sql
-- Start Transaction
START TRANSACTION;
BEGIN;

-- Commit Transaction
COMMIT;

-- Rollback Transaction
ROLLBACK;

-- Savepoint
SAVEPOINT savepoint_name;
ROLLBACK TO savepoint_name;
RELEASE SAVEPOINT savepoint_name;

-- Autocommit
SET AUTOCOMMIT = 0;  -- Disable autocommit
SET AUTOCOMMIT = 1;  -- Enable autocommit
```

## Information Schema Commands

### Database Information
```sql
-- Show Database Information
SELECT * FROM INFORMATION_SCHEMA.SCHEMATA;

-- Show Table Information
SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'database_name';

-- Show Column Information
SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'database_name';

-- Show Index Information
SELECT * FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA = 'database_name';

-- Show Foreign Key Information
SELECT * FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
WHERE CONSTRAINT_SCHEMA = 'database_name';
```

## Utility Commands

### System Information
```sql
-- Show Variables
SHOW VARIABLES;
SHOW VARIABLES LIKE 'version%';

-- Show Status
SHOW STATUS;
SHOW STATUS LIKE 'Connections';

-- Show Processes
SHOW PROCESSLIST;
SHOW FULL PROCESSLIST;

-- Kill Process
KILL process_id;

-- Show Engines
SHOW ENGINES;

-- Show Character Sets
SHOW CHARACTER SET;

-- Show Collations
SHOW COLLATION;
```

### Performance Commands
```sql
-- Explain Query
EXPLAIN SELECT * FROM students WHERE age > 20;
EXPLAIN FORMAT=JSON SELECT * FROM students WHERE age > 20;

-- Analyze Table
ANALYZE TABLE students;

-- Optimize Table
OPTIMIZE TABLE students;

-- Check Table
CHECK TABLE students;

-- Repair Table
REPAIR TABLE students;
```

### Import/Export Commands
```sql
-- Load Data from File
LOAD DATA INFILE 'file_path.csv'
INTO TABLE students
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Select Into Outfile
SELECT * FROM students
INTO OUTFILE 'output.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
```

## Common Data Types

### Numeric Types
```sql
-- Integer Types
TINYINT      -- 1 byte (-128 to 127)
SMALLINT     -- 2 bytes (-32,768 to 32,767)
MEDIUMINT    -- 3 bytes (-8,388,608 to 8,388,607)
INT          -- 4 bytes (-2,147,483,648 to 2,147,483,647)
BIGINT       -- 8 bytes (very large range)

-- Decimal Types
DECIMAL(M,D) -- Exact decimal (M=total digits, D=decimal places)
NUMERIC(M,D) -- Same as DECIMAL
FLOAT(M,D)   -- Single precision floating point
DOUBLE(M,D)  -- Double precision floating point

-- Bit Type
BIT(M)       -- Bit field (M = 1 to 64)
```

### String Types
```sql
-- Fixed Length
CHAR(M)      -- Fixed length string (0 to 255 characters)

-- Variable Length
VARCHAR(M)   -- Variable length string (0 to 65,535 characters)
TEXT         -- Text blob (0 to 65,535 characters)
MEDIUMTEXT   -- Medium text blob (0 to 16,777,215 characters)
LONGTEXT     -- Long text blob (0 to 4,294,967,295 characters)

-- Binary
BINARY(M)    -- Fixed length binary string
VARBINARY(M) -- Variable length binary string
BLOB         -- Binary large object
```

### Date and Time Types
```sql
DATE         -- Date (YYYY-MM-DD)
TIME         -- Time (HH:MM:SS)
DATETIME     -- Date and time (YYYY-MM-DD HH:MM:SS)
TIMESTAMP    -- Timestamp (with timezone)
YEAR         -- Year (YYYY)
```

### JSON Type (MySQL 5.7+)
```sql
-- JSON column
CREATE TABLE products (
    id INT PRIMARY KEY,
    data JSON
);

-- JSON functions
SELECT JSON_EXTRACT(data, '$.name') FROM products;
SELECT data->'$.name' FROM products;  -- Shorthand
SELECT data->>'$.name' FROM products; -- Unquoted
```

This reference covers the most commonly used SQL commands across different categories. Each command includes basic syntax and practical examples to help you understand how to use them effectively.