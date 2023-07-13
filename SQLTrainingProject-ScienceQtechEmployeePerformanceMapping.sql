/*

ScienceQtech Employee Performance Mapping.
Course-end Project 2 

DESCRIPTION
ScienceQtech is a startup that works in the Data Science field. 
ScienceQtech has worked on fraud detection, market basket, self-driving cars, supply chain, algorithmic early detection of lung cancer, 
customer sentiment, and drug discovery field. With the annual appraisal cycle around the corner, the HR department has asked you 
(Junior Database Administrator) to generate reports on employee details, their performance, and on the project that the employees have 
undertaken, to analyze the employee database and extract specific data based on different requirements.
 
Objective: 
To facilitate a better understanding, managers have provided ratings for each employee which will help the HR department to finalize 
the employee performance mapping. As a DBA, you should find the maximum salary of the employees and ensure that all jobs are meeting 
the organization's profile standard. You also need to should also determine whether or not employees need a promotion, and calculate 
bonuses to find extra cost for expenses. This will raise the overall performance of the organization by ensuring that all required 
employees receive training.


*/

-- 1. Create a database named project and employee, then import data_science_team.csv and proj_table.csv into the project database and emp_record_table.csv into the employee database from the given resources.
CREATE DATABASE IF NOT EXISTS project;
CREATE DATABASE IF NOT EXISTS employee;
SHOW DATABASES;


CREATE TABLE project.data_science_team (
EMPLOYEE_ID varchar(4) NOT NULL PRIMARY KEY,
FIRST_NAME varchar(50) NOT NULL,
LAST_NAME varchar(50) NOT NULL,
GENDER varchar(1) NOT NULL,
ROLE varchar(50) NOT NULL,
DEPARTMENT varchar(50) NOT NULL,
EXPERIENCE INT ,
COUNTRY VARCHAR(50),
CONTINENT VARCHAR(50)
);
INSERT INTO project.data_science_team(EMPLOYEE_ID,FIRST_NAME, LAST_NAME, GENDER, ROLE, DEPARTMENT, EXPERIENCE, COUNTRY, CONTINENT ) 
VALUES
("E260","Roy","Collins","M","SENIOR DATA SCIENTIST","RETAIL",7,"INDIA","ASIA"),
("E245","Nian","Zhen","M","SENIOR DATA SCIENTIST","RETAIL",6,"CHINA","ASIA"),
("E620","Katrina","Allen","F","JUNIOR DATA SCIENTIST","RETAIL",2,"INDIA","ASIA"),
("E640","Jenifer","Jhones","F","JUNIOR DATA SCIENTIST","RETAIL",1,"COLOMBIA","SOUTH AMERICA"),
("E403","Steve","Hoffman","M","ASSOCIATE DATA SCIENTIST","FINANCE",4,"USA","NORTH AMERICA"),
("E204","Karene","Nowak","F","SENIOR DATA SCIENTIST","AUTOMOTIVE",8,"GERMANY","EUROPE"),
("E057","Dorothy","Wilson","F","SENIOR DATA SCIENTIST","HEALTHCARE",9,"USA","NORTH AMERICA"),
("E010","William","Butler","M","LEAD DATA SCIENTIST","AUTOMOTIVE",12,"FRANCE","EUROPE"),
("E478","David","Smith","M","ASSOCIATE DATA SCIENTIST","RETAIL",3,"COLOMBIA","SOUTH AMERICA"),
("E005","Eric","Hoffman","M","LEAD DATA SCIENTIST","FINANCE",11,"USA","NORTH AMERICA"),
("E052","Dianna","Wilson","F","SENIOR DATA SCIENTIST","HEALTHCARE",6,"CANADA","NORTH AMERICA"),
("E505","Chad","Wilson","M","ASSOCIATE DATA SCIENTIST","HEALTHCARE",5,"CANADA","NORTH AMERICA"),
("E532","Claire","Brennan","F","ASSOCIATE DATA SCIENTIST","AUTOMOTIVE",3,"GERMANY","EUROPE")
;

CREATE TABLE project.project(

PROJECT_ID varchar(4) NOT NULL PRIMARY KEY,
PROJECT_NAME varchar(50) NOT NULL,
PROJECT_DOMAIN varchar(50) NOT NULL,
START_DATE DATE NOT NULL,  
CLOSURE_DATE DATE NOT NULL,
DEV_QTR varchar(2), 
STATUS varchar(25)
);

INSERT INTO project.project(PROJECT_ID, PROJECT_NAME, PROJECT_DOMAIN, START_DATE, CLOSURE_DATE, DEV_QTR, STATUS)
VALUES
("P103","Drug Discovery","HEALTHCARE",'2021-04-06','2021-06-20',"Q1","DONE"),
("P105","Fraud Detection","FINANCE",'2021-04-11','2021-06-25',"Q1","DONE"),
("P208","Algorithmic Trading","FINANCE",'2022-01-16','2022-03-27',"Q4","YTS"),
("P109","Market Basket Analysis","RETAIL",'2021-04-12','2021-06-30',"Q1","DELAYED"),
("P204","Supply Chain Management","AUTOMOTIVE",'2021-07-15','2021-09-28',"Q2","WIP"),
("P406","Customer Sentiment Analysis","RETAIL",'2021-07-09','2021-09-24',"Q2","WIP"),
("P302","Early Detection of Lung Cancer","HEALTHCARE",'2021-10-08','2021-12-18',"Q3","YTS"),
("P201","Self Driving Cars","AUTOMOTIVE",'2022-01-12','2022-03-30',"Q4","YTS")
;

CREATE TABLE employee.employee_record(
EMPLOYEE_ID varchar(4) NOT NULL PRIMARY KEY,
FIRST_NAME varchar(50) NOT NULL,
LAST_NAME varchar(50) NOT NULL,
GENDER varchar(1) NOT NULL,
ROLE varchar(50) NOT NULL,
DEPARTMENT varchar(50) NOT NULL,
EXPERIENCE INT,
COUNTRY VARCHAR(50),
CONTINENT VARCHAR(50),
SALARY FLOAT,
EMPLOYEE_RATING INT,
MANAGER_ID varchar(4)
);

INSERT INTO employee.employee_record()
VALUES
("E260","Roy","Collins","M","SENIOR DATA SCIENTIST","RETAIL",7,"INDIA","ASIA",7000,3,"E583"),
("E245","Nian","Zhen","M","SENIOR DATA SCIENTIST","RETAIL",6,"CHINA","ASIA",6500,2,"E583"),
("E620","Katrina","Allen","F","JUNIOR DATA SCIENTIST","RETAIL",2,"INDIA","ASIA",3000,1,"E612"),
("E640","Jenifer","Jhones","F","JUNIOR DATA SCIENTIST","RETAIL",1,"COLOMBIA","SOUTH AMERICA",2800,4,"E612"),
("E403","Steve","Hoffman","M","ASSOCIATE DATA SCIENTIST","FINANCE",4,"USA","NORTH AMERICA",5000,3,"E103"),
("E204","Karene","Nowak","F","SENIOR DATA SCIENTIST","AUTOMOTIVE",8,"GERMANY","EUROPE",7500,5,"E428"),
("E057","Dorothy","Wilson","F","SENIOR DATA SCIENTIST","HEALTHCARE",9,"USA","NORTH AMERICA",7700,1,"E083"),
("E010","William","Butler","M","LEAD DATA SCIENTIST","AUTOMOTIVE",12,"FRANCE","EUROPE",9000,2,"E428"),
("E478","David","Smith","M","ASSOCIATE DATA SCIENTIST","RETAIL",3,"COLOMBIA","SOUTH AMERICA",4000,4,"E583"),
("E005","Eric","Hoffman","M","LEAD DATA SCIENTIST","FINANCE",11,"USA","NORTH AMERICA",8500,3,"E103"),
("E052","Dianna","Wilson","F","SENIOR DATA SCIENTIST","HEALTHCARE",6,"CANADA","NORTH AMERICA",5500,5,"E083"),
("E505","Chad","Wilson","M","ASSOCIATE DATA SCIENTIST","HEALTHCARE",5,"CANADA","NORTH AMERICA",5000,2,"E083"),
("E532","Claire","Brennan","F","ASSOCIATE DATA SCIENTIST","AUTOMOTIVE",3,"GERMANY","EUROPE",4300,1,"E428"),
("E083","Patrick","Voltz","M","MANAGER","HEALTHCARE",15,"USA","NORTH AMERICA",9500,5,"E001"),
("E583","Janet","Hale","F","MANAGER","RETAIL",14,"COLOMBIA","SOUTH AMERICA",10000,2,"E001"),
("E103","Emily","Grove","F","MANAGER","FINANCE",14,"CANADA","NORTH AMERICA",10500,4,"E001"),
("E612","Tracy","Norris","F","MANAGER","RETAIL",13,"INDIA","ASIA",8500,4,"E001"),
("E428","Pete","Allen","M","MANAGER","AUTOMOTIVE",14,"GERMANY","EUROPE",11000,4,"E001"),
("E001","Arthur","Black","M","PRESIDENT","ALL",20,"USA","NORTH AMERICA",16500,5,"E001")
;

-- 2. Create an ER diagram for the given project and the employee databases.





-- 3. Write a query to fetch EMP_ID, FIRST_NAME, LAST_NAME, GENDER, and DEPARTMENT from the employee record table, and make a list of employees and details of their department.
SELECT EMPLOYEE_ID, FIRST_NAME,  LAST_NAME, GENDER, DEPARTMENT
FROM employee.employee_record;




/* 4. Write a query to fetch EMP_ID, FIRST_NAME, LAST_NAME, GENDER, DEPARTMENT, and EMP_RATING if the EMP_RATING is:
•	less than two
•	greater than four
•	between two and four */

SELECT EMPLOYEE_ID, FIRST_NAME,  LAST_NAME, GENDER, DEPARTMENT, EMPLOYEE_RATING
FROM employee.employee_record
WHERE EMPLOYEE_RATING < 2;

SELECT EMPLOYEE_ID, FIRST_NAME,  LAST_NAME, GENDER, DEPARTMENT, EMPLOYEE_RATING
FROM employee.employee_record
WHERE EMPLOYEE_RATING > 4;

SELECT EMPLOYEE_ID, FIRST_NAME,  LAST_NAME, GENDER, DEPARTMENT, EMPLOYEE_RATING
FROM employee.employee_record
WHERE EMPLOYEE_RATING BETWEEN 2 AND 4;



-- 5. Write a query to concatenate the FIRST_NAME and the LAST_NAME of employees in the Finance department from the employee table and then give the resultant column alias as NAME.
SELECT 
    EMPLOYEE_ID,
    CONCAT(FIRST_NAME, ' ', LAST_NAME) AS NAME,
    GENDER,
    DEPARTMENT,
    EMPLOYEE_RATING
FROM
    employee.employee_record
WHERE
    UPPER(DEPARTMENT) = 'FINANCE';

-- 6. Write a query to list only those employees who have someone reporting to them. Also, show the number of reporters (includeing the President and the CEO of the organization).
-- practicing Common Table Expressions (CTE)
-- PL-people leaders, DRS-direct reports summary

WITH PL AS(SELECT EMPLOYEE_ID, ROLE, FIRST_NAME, LAST_NAME FROM employee.employee_record WHERE ROLE in('MANAGER','PRESIDENT')),
	DRS AS(SELECT manager_id, count(DISTINCT employee_id) AS DIRECT_REPORTS_N FROM employee.employee_record WHERE EMPLOYEE_ID != MANAGER_ID GROUP BY MANAGER_ID)
SELECT EMPLOYEE_ID, ROLE, FIRST_NAME, LAST_NAME, DIRECT_REPORTS_N 
FROM PL
JOIN DRS 
WHERE PL.EMPLOYEE_ID = DRS.MANAGER_ID
ORDER BY DIRECT_REPORTS_N DESC
;
-- 7. Write a query to list down all the employees from the healthcare and finance departments using union. Take data from the employee record table.
WITH H AS(SELECT * FROM employee.employee_record WHERE UPPER(DEPARTMENT) = 'HEALTHCARE'),
	F AS(SELECT * FROM employee.employee_record WHERE UPPER(DEPARTMENT) = 'FINANCE')
    SELECT * FROM H  
    UNION
    SELECT * FROM F;

/* 8. Write a query to list down employee details such as EMP_ID, FIRST_NAME, LAST_NAME, ROLE, DEPARTMENT, and EMP_RATING grouped by dept. 
Also include the respective employee rating along with the max emp rating for the department. */
WITH E AS(SELECT EMPLOYEE_ID, FIRST_NAME, LAST_NAME, ROLE, DEPARTMENT, EMPLOYEE_RATING FROM employee.employee_record ),
	MR AS(SELECT DEPARTMENT, MAX(EMPLOYEE_RATING) AS DEPARTMENT_MAX_RATING FROM employee.employee_record GROUP BY DEPARTMENT)
SELECT EMPLOYEE_ID, FIRST_NAME, LAST_NAME, ROLE, E.DEPARTMENT, EMPLOYEE_RATING, DEPARTMENT_MAX_RATING
FROM E
JOIN MR 
WHERE E.DEPARTMENT = MR.DEPARTMENT
ORDER BY E.DEPARTMENT, ROLE, EMPLOYEE_ID;

-- 9. Write a query to calculate the minimum and the maximum salary of the employees in each role. Take data from the employee record table.
SELECT ROLE, MAX(SALARY) AS MAX_SALARY, MIN(SALARY) AS MIN_SALARY FROM employee.employee_record GROUP BY ROLE;

-- 10. Write a query to assign ranks to each employee based on their experience. Take data from the employee record table.
/*RANK() OVER(PARTITION BY <expr1>*/
SELECT EMPLOYEE_ID, CONCAT(FIRST_NAME, " ", LAST_NAME) AS FULL_NAME, EXPERIENCE, RANK() OVER(ORDER BY EXPERIENCE DESC) AS EXPERIENCE_RANKING FROM employee.employee_record;

-- 11. Write a query to create a view that displays employees in various countries whose salary is more than six thousand. Take data from the employee record table.
CREATE VIEW employee.SALARIES_BY_COUNTRY AS
(SELECT EMPLOYEE_ID, CONCAT(FIRST_NAME, " ", LAST_NAME) AS FULL_NAME, SALARY, COUNTRY
	FROM employee.employee_record
    WHERE SALARY > 6000
    ORDER BY COUNTRY, SALARY DESC);

select * from employee.SALARIES_BY_COUNTRY;

-- 12. Write a nested query to find employees with experience of more than ten years. Take data from the employee record table.
SELECT * 
FROM employee.employee_record
WHERE EMPLOYEE_ID IN(SELECT EMPLOYEE_ID FROM employee.employee_record WHERE EXPERIENCE > 10)
ORDER BY EXPERIENCE DESC;

/* 13. Write a query to create a stored procedure to retrieve the details of the employees 
whose experience is more than three years. Take data from the employee record table.*/
DELIMITER &&
CREATE PROCEDURE employee.EMPLOYEES_WITH_GREATER_THAN_3_YEARS_EXPERIENCE()
BEGIN
SELECT * FROM employee.employee_record
where EXPERIENCE > 3;
END && DELIMITER &&;
CALL employee.EMPLOYEES_WITH_GREATER_THAN_3_YEARS_EXPERIENCE();
DELIMITER ;
/* 14. Write a query using stored functions in the project table to check whether the job profile 
assigned to each employee in the data science team matches the organization’s set standard.
     The standard being:
For an employee with experience less than or equal to 2 years assign 'JUNIOR DATA SCIENTIST',
For an employee with the experience of 2 to 5 years assign 'ASSOCIATE DATA SCIENTIST',
For an employee with the experience of 5 to 10 years assign 'SENIOR DATA SCIENTIST',
For an employee with the experience of 10 to 12 years assign 'LEAD DATA SCIENTIST',
For an employee with the experience of 12 to 16 years assign 'MANAGER'. */

DELIMITER &&
-- DROP FUNCTION project.STANDARD_ASSESSMENT;
CREATE FUNCTION project.STANDARD_ASSESSMENT(exp INT)
RETURNS VARCHAR(500) DETERMINISTIC
BEGIN DECLARE role_standard varchar(500);
IF exp <= 2 THEN SET role_standard = 'JUNIOR DATA SCIENTIST';
ELSEIF exp BETWEEN 3 AND 5 THEN SET role_standard = 'ASSOCIATE DATA SCIENTIST';
ELSEIF exp BETWEEN 6 AND 10 THEN SET role_standard = 'SENIOR DATA SCIENTIST';
ELSEIF exp BETWEEN 11 AND 12 THEN SET role_standard = 'LEAD DATA SCIENTIST';
ELSEIF exp BETWEEN 13 AND 16 THEN SET role_standard = 'MANAGER';
END IF; RETURN (role_standard);
END &&
DELIMITER &&;

SELECT EMPLOYEE_ID, FIRST_NAME, LAST_NAME,  EXPERIENCE, ROLE, project.STANDARD_ASSESSMENT(EXPERIENCE) AS ROLE_STANDARD 
	FROM project.data_science_team;
DELIMITER ;

 
-- 15. Create an index to improve the cost and performance of the query to find the employee whose FIRST_NAME is ‘Eric’ in the employee table after checking the execution plan.
SELECT * FROM employee.employee_record
WHERE FIRST_NAME = 'Eric';
CREATE INDEX idx_first_name ON employee.employee_record(FIRST_NAME);
SELECT * FROM employee.employee_record
WHERE FIRST_NAME = 'Eric';

-- 16. Write a query to calculate the bonus for all the employees, based on their ratings and salaries (Use the formula: 5% of salary * employee rating).
SELECT 
*, .05*SALARY*EMPLOYEE_RATING AS BONUS
FROM employee.employee_record;

-- 17. Write a query to calculate the average salary distribution based on the continent and country. Take data from the employee record table.
SELECT CONTINENT, COUNTRY, ROUND(AVG(SALARY),0) AS AVG_SALARY 
FROM employee.employee_record
GROUP BY CONTINENT, COUNTRY
ORDER BY CONTINENT, COUNTRY;









