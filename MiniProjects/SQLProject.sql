/* Welcome to the SQL mini project. For this project, you will use
Springboard' online SQL platform, which you can log into through the
following link:

https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

Note that, if you need to, you can also download these tables locally.

In the mini project, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */



/* Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do. */
SELECT name as "Facility"
FROM  Facilities 
WHERE Facilities.membercost > 0

/* Q2: How many facilities do not charge a fee to members? */
SELECT count(*) as "Facilities with Nocharge"
FROM  Facilities 
WHERE Facilities.membercost = 0

/* Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */
SELECT 	Facilities.facid, 
	Facilities.name as "Facility", 
	Facilities.membercost as "Member Cost", 
	Facilities.monthlymaintenance as "Monthly Maintenance"
FROM  Facilities
WHERE Facilities.membercost / Facilities.monthlymaintenance < .20
AND   Facilities.membercost > 0

/* Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator. */
SELECT  Facilities.facid, 
	Facilities.guestcost as "Guest Cost",
	Facilities.initialoutlay as "Initial Outlay",
	Facilities.membercost as "Member Cost",
	Facilities.monthlymaintenance as "Monthly Maintenance",
	Facilities.name as "Facility Name"
FROM  Facilities
WHERE facid in (1,5)


/* Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question. */
SELECT 	Facilities.name as "Facility",
	Facilities.monthlymaintenance as "Monthly Maintenance",
	CASE WHEN Facilities.monthlymaintenance <= 100 THEN 'cheap'
	     ELSE 'expensive'
	END as "Cost Classification"
FROM Facilities

/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution. */
SELECT Members.firstname as "First Name", Members.surname as "Last Name"
FROM  Members
WHERE joindate in (SELECT MAX(joindate) FROM Members)

/* Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */
SELECT 	DISTINCT
	Facilities.name as "Facility Name",
	concat(Members.surname, ', ', Members.firstname) as "Member Name"
FROM  Members
JOIN  Bookings
  ON  Members.memid = Bookings.memid
JOIN  Facilities
  ON  Facilities.facid = Bookings.facid
WHERE name like 'Tennis Court%'
ORDER BY 2,1

/* Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */
SELECT f1.name as "Facility Name",
       concat(m.surname,', ',m.firstname) as "Member Name", 
       CASE WHEN b1.memid = 0 THEN f1.guestcost * b1.slots
                ELSE f1.membercost * b1.slots
            END as "Cost"
FROM  Bookings b1
JOIN  Bookings b2
  ON  b1.memid = b2.memid 
JOIN  Facilities f2
  ON  f2.facid = b2.facid
JOIN  Facilities f1
  ON  f1.facid = b1.facid
JOIN  Members m
  ON  m.memid = b1.memid
WHERE DATE(b1.starttime) = '2012-09-14'
AND   DATE(b2.starttime) = '2012-09-14'
group by b1.bookid, 
         CASE WHEN b1.memid = 0 THEN f1.guestcost * b1.slots
              ELSE f1.membercost * b1.slots
         END 
HAVING   SUM(CASE WHEN b2.memid = 0 THEN f2.guestcost * b2.slots
                  ELSE f2.membercost * b2.slots
             END) > 30
ORDER BY 3 DESC, 2, 1

/* Q9: This time, produce the same result as in Q8, but using a subquery. */
SELECT Facilities.name as "Facility Name", 
       CONCAT(Members.surname,', ',Members.firstname) as "Member Name", 
       CASE WHEN Members.memid = 0 THEN Facilities.guestcost * Bookings.slots
            ELSE Facilities.membercost * Bookings.slots
       END as "Cost"
FROM  Bookings
JOIN  Members
  ON  Bookings.memid = Members.memid
JOIN  Facilities
  ON  Bookings.facid = Facilities.facid
WHERE DATE(Bookings.starttime) = '2012-09-14'
AND   Members.memid in (
			SELECT Bookings.memid as memid 
			FROM  Bookings
			JOIN  Facilities
			  ON  Facilities.facid = Bookings.facid
			WHERE DATE(Bookings.starttime) = '2012-09-14'
			GROUP BY 1
			HAVING SUM(CASE WHEN Bookings.memid = 0 THEN Facilities.guestcost * slots
                 		        ELSE Facilities.membercost * slots
	    		           END) > 30
                       ) 
GROUP BY Bookings.bookid
ORDER BY 3 DESC, 1, 2

/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */
SELECT Facilities.name as "Facility Name", 
       SUM(CASE WHEN Bookings.memid = 0 THEN guestcost * slots
	        ELSE membercost * slots
           END) as " Total Revenue"
FROM  Bookings
JOIN  Facilities
  ON  Bookings.facid = Facilities.facid
GROUP BY 1
HAVING SUM(CASE WHEN Bookings.memid = 0 THEN guestcost * slots
	        ELSE membercost * slots
           END) < 1000
ORDER BY 2
