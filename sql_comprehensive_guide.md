# Complete SQL Guide with Code Examples

## SQL Fundamentals

### 1. Database Concepts
SQL (Structured Query Language) is a programming language designed for managing relational databases. Key concepts include:
- **Tables**: Store data in rows and columns
- **Primary Key**: Unique identifier for each row
- **Foreign Key**: Links tables together
- **Schema**: Database structure definition

### 2. MySQL Setup
```sql
-- Check MySQL version
SELECT VERSION();

-- Show available databases
SHOW DATABASES;

-- Create a new database
CREATE DATABASE company_db;

-- Use the database
USE company_db;
```

### 3. CREATE/ALTER Tables
```sql
-- Create a table
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE,
    salary DECIMAL(10,2),
    department_id INT
);

-- Create departments table
CREATE TABLE departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    department_name VARCHAR(50) NOT NULL,
    location VARCHAR(100)
);

-- ALTER table to add foreign key
ALTER TABLE employees 
ADD CONSTRAINT fk_department 
FOREIGN KEY (department_id) REFERENCES departments(department_id);

-- Add a new column
ALTER TABLE employees ADD COLUMN phone VARCHAR(15);

-- Modify column
ALTER TABLE employees MODIFY COLUMN salary DECIMAL(12,2);

-- Drop column
ALTER TABLE employees DROP COLUMN phone;
```

### 4. Basic Queries (SELECT)
```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, salary FROM employees;

-- Select with alias
SELECT first_name AS 'First Name', 
       last_name AS 'Last Name',
       salary AS 'Annual Salary'
FROM employees;

-- Select distinct values
SELECT DISTINCT department_id FROM employees;
```

## SQL Data Manipulation

### 1. INSERT/UPDATE/DELETE
```sql
-- Insert single record
INSERT INTO departments (department_name, location)
VALUES ('Human Resources', 'New York');

-- Insert multiple records
INSERT INTO employees (first_name, last_name, email, hire_date, salary, department_id)
VALUES 
    ('John', 'Doe', 'john.doe@company.com', '2023-01-15', 65000.00, 1),
    ('Jane', 'Smith', 'jane.smith@company.com', '2023-02-20', 70000.00, 2),
    ('Mike', 'Johnson', 'mike.johnson@company.com', '2023-03-10', 55000.00, 1);

-- UPDATE records
UPDATE employees 
SET salary = 68000.00 
WHERE employee_id = 1;

-- UPDATE multiple columns
UPDATE employees 
SET salary = salary * 1.1, 
    email = 'john.doe.updated@company.com'
WHERE first_name = 'John';

-- DELETE records
DELETE FROM employees 
WHERE employee_id = 3;

-- DELETE with condition
DELETE FROM employees 
WHERE salary < 60000;
```

### 2. Filtering (WHERE)
```sql
-- Basic WHERE conditions
SELECT * FROM employees WHERE salary > 60000;

-- Multiple conditions with AND
SELECT * FROM employees 
WHERE salary > 60000 AND department_id = 1;

-- Multiple conditions with OR
SELECT * FROM employees 
WHERE department_id = 1 OR department_id = 2;

-- NOT condition
SELECT * FROM employees 
WHERE NOT department_id = 1;

-- IN operator
SELECT * FROM employees 
WHERE department_id IN (1, 2, 3);

-- BETWEEN operator
SELECT * FROM employees 
WHERE salary BETWEEN 50000 AND 70000;

-- IS NULL / IS NOT NULL
SELECT * FROM employees 
WHERE email IS NOT NULL;
```

### 3. Sorting (ORDER BY)
```sql
-- Sort ascending (default)
SELECT * FROM employees ORDER BY salary;

-- Sort descending
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple column sorting
SELECT * FROM employees 
ORDER BY department_id ASC, salary DESC;

-- Sort by alias
SELECT first_name, last_name, salary AS annual_salary
FROM employees 
ORDER BY annual_salary DESC;
```

### 4. LIMIT Clauses
```sql
-- Limit number of results
SELECT * FROM employees LIMIT 5;

-- Skip rows and limit (OFFSET)
SELECT * FROM employees LIMIT 5 OFFSET 2;

-- Alternative syntax for pagination
SELECT * FROM employees LIMIT 2, 5;  -- Skip 2, take 5

-- Top N records
SELECT * FROM employees 
ORDER BY salary DESC 
LIMIT 3;
```

## Advanced SQL Queries

### 1. Aggregate Functions (COUNT, SUM)
```sql
-- COUNT function
SELECT COUNT(*) AS total_employees FROM employees;
SELECT COUNT(DISTINCT department_id) AS unique_departments FROM employees;

-- SUM function
SELECT SUM(salary) AS total_payroll FROM employees;

-- AVG function
SELECT AVG(salary) AS average_salary FROM employees;

-- MIN and MAX
SELECT MIN(salary) AS lowest_salary, 
       MAX(salary) AS highest_salary 
FROM employees;

-- Multiple aggregates
SELECT 
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    MIN(hire_date) AS earliest_hire,
    MAX(hire_date) AS latest_hire
FROM employees;
```

### 2. GROUP BY/HAVING
```sql
-- Basic GROUP BY
SELECT department_id, COUNT(*) AS employee_count
FROM employees 
GROUP BY department_id;

-- GROUP BY with aggregate functions
SELECT department_id, 
       COUNT(*) AS employee_count,
       AVG(salary) AS avg_salary,
       SUM(salary) AS total_salary
FROM employees 
GROUP BY department_id;

-- HAVING clause (filter groups)
SELECT department_id, COUNT(*) AS employee_count
FROM employees 
GROUP BY department_id
HAVING COUNT(*) > 2;

-- HAVING with multiple conditions
SELECT department_id, AVG(salary) AS avg_salary
FROM employees 
GROUP BY department_id
HAVING AVG(salary) > 60000 AND COUNT(*) >= 2;
```

### 3. Wildcard Operators
```sql
-- LIKE with % (zero or more characters)
SELECT * FROM employees 
WHERE first_name LIKE 'J%';  -- Starts with J

SELECT * FROM employees 
WHERE email LIKE '%@company.com';  -- Ends with @company.com

-- LIKE with _ (single character)
SELECT * FROM employees 
WHERE first_name LIKE 'J_hn';  -- J followed by any char, then hn

-- Case-insensitive search
SELECT * FROM employees 
WHERE LOWER(first_name) LIKE 'j%';

-- Multiple patterns
SELECT * FROM employees 
WHERE first_name LIKE 'J%' OR first_name LIKE 'M%';
```

### 4. Subqueries
```sql
-- Subquery in WHERE clause
SELECT * FROM employees 
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Subquery with IN
SELECT * FROM employees 
WHERE department_id IN (
    SELECT department_id FROM departments 
    WHERE location = 'New York'
);

-- Correlated subquery
SELECT e1.first_name, e1.last_name, e1.salary
FROM employees e1
WHERE e1.salary > (
    SELECT AVG(e2.salary) 
    FROM employees e2 
    WHERE e2.department_id = e1.department_id
);

-- EXISTS subquery
SELECT * FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e 
    WHERE e.department_id = d.department_id
);
```

## SQL Joins

### 1. INNER JOIN
```sql
-- Basic INNER JOIN
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- INNER JOIN with WHERE
SELECT e.first_name, e.last_name, d.department_name, e.salary
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
WHERE e.salary > 60000;

-- Multiple table JOIN
CREATE TABLE projects (
    project_id INT PRIMARY KEY,
    project_name VARCHAR(100),
    department_id INT
);

SELECT e.first_name, d.department_name, p.project_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
INNER JOIN projects p ON d.department_id = p.department_id;
```

### 2. LEFT/RIGHT JOIN
```sql
-- LEFT JOIN (all records from left table)
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- RIGHT JOIN (all records from right table)
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;

-- Find employees without departments
SELECT e.first_name, e.last_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id
WHERE d.department_id IS NULL;
```

### 3. FULL OUTER JOIN
```sql
-- FULL OUTER JOIN (MySQL doesn't support directly, use UNION)
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id
UNION
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;
```

### 4. Self Joins
```sql
-- Add manager_id to employees table
ALTER TABLE employees ADD COLUMN manager_id INT;

-- Self join to find employees and their managers
SELECT 
    e.first_name AS employee_name,
    m.first_name AS manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;
```

## SQL Functions

### 1. String Functions
```sql
-- CONCAT
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM employees;

-- LENGTH
SELECT first_name, LENGTH(first_name) AS name_length FROM employees;

-- UPPER/LOWER
SELECT UPPER(first_name) AS upper_name, LOWER(last_name) AS lower_name FROM employees;

-- SUBSTRING
SELECT SUBSTRING(email, 1, 5) AS email_prefix FROM employees;

-- REPLACE
SELECT REPLACE(email, '@company.com', '@newcompany.com') AS new_email FROM employees;

-- TRIM
SELECT TRIM('  John Doe  ') AS trimmed_name;

-- LEFT/RIGHT
SELECT LEFT(first_name, 3) AS first_three, RIGHT(last_name, 3) AS last_three FROM employees;
```

### 2. Date Functions
```sql
-- Current date and time
SELECT NOW(), CURDATE(), CURTIME();

-- Date formatting
SELECT DATE_FORMAT(hire_date, '%Y-%m-%d') AS formatted_date FROM employees;
SELECT DATE_FORMAT(hire_date, '%M %d, %Y') AS readable_date FROM employees;

-- Date arithmetic
SELECT first_name, hire_date, 
       DATEDIFF(NOW(), hire_date) AS days_employed
FROM employees;

-- Extract parts of date
SELECT 
    YEAR(hire_date) AS hire_year,
    MONTH(hire_date) AS hire_month,
    DAY(hire_date) AS hire_day
FROM employees;

-- Add/subtract dates
SELECT DATE_ADD(hire_date, INTERVAL 1 YEAR) AS one_year_later FROM employees;
SELECT DATE_SUB(NOW(), INTERVAL 30 DAY) AS thirty_days_ago;
```

### 3. Numeric Functions
```sql
-- ROUND, CEIL, FLOOR
SELECT salary, 
       ROUND(salary, 0) AS rounded_salary,
       CEIL(salary/1000) AS ceiling_thousands,
       FLOOR(salary/1000) AS floor_thousands
FROM employees;

-- ABS, POWER, SQRT
SELECT ABS(-100) AS absolute_value,
       POWER(2, 3) AS power_result,
       SQRT(16) AS square_root;

-- MOD (modulo)
SELECT employee_id, MOD(employee_id, 2) AS is_even FROM employees;

-- GREATEST/LEAST
SELECT GREATEST(10, 20, 5) AS max_value,
       LEAST(10, 20, 5) AS min_value;
```

### 4. Stored Procedures
```sql
-- Create a simple stored procedure
DELIMITER //
CREATE PROCEDURE GetEmployeesByDepartment(IN dept_id INT)
BEGIN
    SELECT * FROM employees WHERE department_id = dept_id;
END //
DELIMITER ;

-- Call the stored procedure
CALL GetEmployeesByDepartment(1);

-- Procedure with OUT parameter
DELIMITER //
CREATE PROCEDURE GetEmployeeCount(IN dept_id INT, OUT emp_count INT)
BEGIN
    SELECT COUNT(*) INTO emp_count 
    FROM employees 
    WHERE department_id = dept_id;
END //
DELIMITER ;

-- Call procedure with OUT parameter
CALL GetEmployeeCount(1, @count);
SELECT @count AS employee_count;

-- Drop procedure
DROP PROCEDURE IF EXISTS GetEmployeesByDepartment;
```

## SQL Optimization

### 1. Indexes
```sql
-- Create index on single column
CREATE INDEX idx_salary ON employees(salary);

-- Create composite index
CREATE INDEX idx_dept_salary ON employees(department_id, salary);

-- Create unique index
CREATE UNIQUE INDEX idx_email ON employees(email);

-- Show indexes
SHOW INDEX FROM employees;

-- Drop index
DROP INDEX idx_salary ON employees;

-- Analyze query performance
EXPLAIN SELECT * FROM employees WHERE salary > 60000;
```

### 2. Query Optimization
```sql
-- Use EXPLAIN to analyze queries
EXPLAIN SELECT e.first_name, d.department_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE e.salary > 60000;

-- Optimize with proper indexing
CREATE INDEX idx_salary ON employees(salary);
CREATE INDEX idx_dept_id ON employees(department_id);

-- Use LIMIT when possible
SELECT * FROM employees ORDER BY salary DESC LIMIT 10;

-- Avoid SELECT * in production
SELECT first_name, last_name, salary FROM employees;

-- Use EXISTS instead of IN for better performance
SELECT * FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.department_id);
```

### 3. Views
```sql
-- Create a view
CREATE VIEW employee_details AS
SELECT 
    e.employee_id,
    CONCAT(e.first_name, ' ', e.last_name) AS full_name,
    e.email,
    e.salary,
    d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- Use the view
SELECT * FROM employee_details WHERE salary > 60000;

-- Update view
CREATE OR REPLACE VIEW employee_details AS
SELECT 
    e.employee_id,
    CONCAT(e.first_name, ' ', e.last_name) AS full_name,
    e.email,
    e.salary,
    d.department_name,
    e.hire_date
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- Drop view
DROP VIEW employee_details;
```

### 4. Transactions
```sql
-- Start transaction
START TRANSACTION;

-- Multiple operations
INSERT INTO departments (department_name, location) VALUES ('IT', 'Boston');
INSERT INTO employees (first_name, last_name, email, department_id) 
VALUES ('Alice', 'Johnson', 'alice@company.com', LAST_INSERT_ID());

-- Commit if all successful
COMMIT;

-- Example with rollback
START TRANSACTION;

UPDATE employees SET salary = salary * 1.2 WHERE department_id = 1;

-- Check if update affected too many rows
SELECT ROW_COUNT();

-- Rollback if something went wrong
ROLLBACK;

-- Using savepoints
START TRANSACTION;

SAVEPOINT sp1;
INSERT INTO employees (first_name, last_name) VALUES ('Test', 'User');

SAVEPOINT sp2;
UPDATE employees SET salary = 100000 WHERE first_name = 'Test';

-- Rollback to specific savepoint
ROLLBACK TO sp1;

COMMIT;
```

## SQL Project Example

### 1. Database Design
```sql
-- Complete database schema for a company management system
CREATE DATABASE company_management;
USE company_management;

-- Departments table
CREATE TABLE departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    department_name VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    budget DECIMAL(12,2)
);

-- Employees table
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE,
    salary DECIMAL(10,2),
    department_id INT,
    manager_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
);

-- Projects table
CREATE TABLE projects (
    project_id INT PRIMARY KEY AUTO_INCREMENT,
    project_name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12,2),
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- Employee-Project junction table
CREATE TABLE employee_projects (
    employee_id INT,
    project_id INT,
    role VARCHAR(50),
    hours_allocated INT,
    PRIMARY KEY (employee_id, project_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);
```

### 2. Complex Queries
```sql
-- Find top performing departments by average salary
SELECT 
    d.department_name,
    COUNT(e.employee_id) AS employee_count,
    AVG(e.salary) AS avg_salary,
    SUM(e.salary) AS total_payroll
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id, d.department_name
ORDER BY avg_salary DESC;

-- Find employees working on multiple projects
SELECT 
    CONCAT(e.first_name, ' ', e.last_name) AS employee_name,
    COUNT(ep.project_id) AS project_count,
    GROUP_CONCAT(p.project_name) AS projects
FROM employees e
JOIN employee_projects ep ON e.employee_id = ep.employee_id
JOIN projects p ON ep.project_id = p.project_id
GROUP BY e.employee_id
HAVING project_count > 1;

-- Hierarchical query for management structure
WITH RECURSIVE management_hierarchy AS (
    SELECT employee_id, first_name, last_name, manager_id, 0 as level
    FROM employees 
    WHERE manager_id IS NULL
    
    UNION ALL
    
    SELECT e.employee_id, e.first_name, e.last_name, e.manager_id, mh.level + 1
    FROM employees e
    JOIN management_hierarchy mh ON e.manager_id = mh.employee_id
)
SELECT * FROM management_hierarchy ORDER BY level, employee_id;
```

### 3. Data Analysis
```sql
-- Monthly hiring trends
SELECT 
    YEAR(hire_date) AS hire_year,
    MONTH(hire_date) AS hire_month,
    COUNT(*) AS hires_count
FROM employees
GROUP BY YEAR(hire_date), MONTH(hire_date)
ORDER BY hire_year, hire_month;

-- Salary analysis by department
SELECT 
    d.department_name,
    MIN(e.salary) AS min_salary,
    MAX(e.salary) AS max_salary,
    AVG(e.salary) AS avg_salary,
    STDDEV(e.salary) AS salary_stddev
FROM departments d
JOIN employees e ON d.department_id = e.department_id
GROUP BY d.department_id, d.department_name;

-- Project budget utilization
SELECT 
    p.project_name,
    p.budget AS allocated_budget,
    SUM(e.salary * ep.hours_allocated / 2080) AS estimated_cost,
    (p.budget - SUM(e.salary * ep.hours_allocated / 2080)) AS remaining_budget
FROM projects p
JOIN employee_projects ep ON p.project_id = ep.project_id
JOIN employees e ON ep.employee_id = e.employee_id
GROUP BY p.project_id, p.project_name, p.budget;
```

### 4. Reporting
```sql
-- Executive dashboard view
CREATE VIEW executive_dashboard AS
SELECT 
    'Total Employees' AS metric,
    COUNT(*) AS value,
    NULL AS department
FROM employees
UNION ALL
SELECT 
    'Average Salary' AS metric,
    ROUND(AVG(salary), 2) AS value,
    NULL AS department
FROM employees
UNION ALL
SELECT 
    'Total Departments' AS metric,
    COUNT(*) AS value,
    NULL AS department
FROM departments
UNION ALL
SELECT 
    'Active Projects' AS metric,
    COUNT(*) AS value,
    NULL AS department
FROM projects
WHERE end_date > CURDATE() OR end_date IS NULL;

-- Department performance report
CREATE VIEW department_performance AS
SELECT 
    d.department_name,
    COUNT(e.employee_id) AS employee_count,
    ROUND(AVG(e.salary), 2) AS avg_salary,
    COUNT(p.project_id) AS active_projects,
    ROUND(AVG(p.budget), 2) AS avg_project_budget
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
LEFT JOIN projects p ON d.department_id = p.department_id
GROUP BY d.department_id, d.department_name;

-- Use the reports
SELECT * FROM executive_dashboard;
SELECT * FROM department_performance ORDER BY employee_count DESC;
```

## Best Practices Summary

1. **Always use proper data types** for optimal storage and performance
2. **Create indexes** on frequently queried columns
3. **Use meaningful table and column names**
4. **Normalize your database** to reduce redundancy
5. **Use transactions** for data integrity
6. **Backup your data** regularly
7. **Test queries** with EXPLAIN before production
8. **Use parameterized queries** to prevent SQL injection
9. **Document your database schema**
10. **Monitor query performance** regularly