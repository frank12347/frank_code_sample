drop table CLASSES;
drop type STUDENT_NT;
drop type PROFESSOR_VA;
drop type PROFESSOR_TY;
drop type STUDENT_TY;

drop table STUDENT



-----------VARRAY
-- create a new professor data type
create or replace type PROFESSOR_TY as object(
first varchar(25),
last varchar(25));

-- now create a varray of professors
create or replace type PROFESSOR_VA as varray(3) of PROFESSOR_TY;


-- create a classes table
create table CLASSES(
ClassName Varchar(15),
Professors PROFESSOR_VA);

describe CLASSES;

insert into CLASSES(ClassName,Professors)
values
('CS669',
PROFESSOR_VA(
  PROFESSOR_TY('Warren','Mansuer'),
  PROFESSOR_TY('Jack','Polnar')));

insert into CLASSES(ClassName, Professors)
values
('CS779',
PROFESSOR_VA(
  PROFESSOR_TY('Bob','Schudy'),
  PROFESSOR_TY('Jack','Polnar')));
  
  
  insert into CLASSES(ClassName)
values
('CS655');
  
  
  delete from CLASSES

  select * from CLASSES
  
  select * from CLASSES C
where C.Professors is NULL;
  
   select C.ClassName, P.* from CLASSES C , TABLE(C.Professors) P;


    select C.ClassName from CLASSES C;
 commit; 
  
  insert into CLASSES(ClassName,Professors)
values
('CS682',
PROFESSOR_VA(
  PROFESSOR_TY('Jack','Polnar')));
  
  

-- let's add some students
-- add student ADT
create or replace type STUDENT_TY as object(
first varchar(25),
last varchar(25),
BUID char (9));

-- create a nested table of students
create or replace type STUDENT_NT as table of STUDENT_TY;

describe CLASSES;

   select C.ClassName, P.* from CLASSES C , TABLE(C.Professors) P;

-- alter our table to have students

alter table CLASSES
  add Students STUDENT_NT
  nested table STUDENTS store as STUDENTS_NT_TAB;


select * from STUDENTS;
select * from STUDENTS_NT_TAB;
select * from CLASSES;


delete from CLASSES
where ClassName='CS779';

describe CLASSES;
  
insert into CLASSES  (ClassName,Professors,Students)  
values
('CS779',
PROFESSOR_VA(
  PROFESSOR_TY('Bob','Schudy'),
  PROFESSOR_TY('Jack','Polnar')),
STUDENT_NT(
  STUDENT_TY('John','Smith','ST1'),
  STUDENT_TY('Mary','Johnson','ST2'),
  STUDENT_TY('Robert','Dole','ST3'))) ;
  
  
  select * from CLASSES

  select C.ClassName, P.*, S.* 
  from CLASSES C, TABLE(C.Professors) P, TABLE(C.Students) S ; 
  
    select distinct S.LAST as StudentLast
  from CLASSES C, TABLE(C.Professors) P, TABLE(C.Students) S ; 
  
    select C.ClassName, P.* 
  from CLASSES C, TABLE(C.Professors) P; 
  
    select C.ClassName, P.*, S.* 
  from CLASSES C, TABLE(C.Professors) P, TABLE(C.Students) S ; 

  --- Updating?
 -- Not working quite yet!
 
  update CLASSES
set Students=
STUDENT_NT(
  STUDENT_TY('John','Smith','ST1'))
where ClassName='CS779';

-- notice how all other students are gone
    select C.ClassName, P.*, S.* 
  from CLASSES C, TABLE(C.Professors) P, TABLE(C.Students) S ; 




  update S
set Students=
STUDENT_NT(
  STUDENT_TY('John','Smith II','ST1'))
  from
   from CLASSES C, TABLE(C.Professors) P, TABLE(C.Students) S ;
where 
S.BUID='ST1'
  
ClassName='CS779';

  select C.ClassName, P.*, S.* 
  from CLASSES C, TABLE(C.Professors) P, TABLE(C.Students) S ; 


  update CLASSES
set Students=
STUDENT_NT(
  STUDENT_TY('John','Smith II','ST1'))
where STUDENT_NT(
  STUDENT_TY('John','Smith','ST1'));


  
select C.ClassName, S.* 
from CLASSES C, TABLE(C.Students) S;

select C.ClassName, S.First, S.Last, S.BUID
from CLASSES C, TABLE(C.Students) S
where S.Last='Smith';

--- not working, play around with next time
update CLASSES C
set S.Last='Smith1'
from CLASSES C, TABLE(C.Students) S
where S.Last='Smith';

update TABLE(select Student from Students where Vehicle_Identification_Number=102) S
set 

update TABLE(select auto_owner from automobile where Vehicle_Identification_Number=102 ) n
SET n.owner.first_name='Maxim' 
where n.owner.first_name='Max';


  update CLASSES C
set Students=
STUDENT_NT(
  STUDENT_TY('John','Smith','ST1'))
where C.Students.Last='Smith';

  update CLASSES
set Students=
STUDENT_NT(
  STUDENT_TY('John','Smith II','ST1'))
where STUDENT_NT(
  STUDENT_TY('John','Smith','ST1'));  
  
  
  


