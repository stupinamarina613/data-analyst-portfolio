-- Задание 2.1: Динамика платежей в Поволжье
SELECT DATE(SUBSTR(time_payment, 7, 4) || '-' || 
       	   SUBSTR(time_payment, 4, 2) || '-' || 
       	   SUBSTR(time_payment, 1, 2)) as day_pay
               , SUM(amt_payment) as sum_pay
FROM payments t1
	JOIN client_info t2 
    		ON t1.id_client = t2.id_client
	JOIN city_info t3
    		ON t2.id_city = t3.id_city
WHERE name_region = 'Поволжье'
GROUP BY day_pay
ORDER BY day_pay;


-- Задание 2.2: Доля мужчин по городам
Select    name_city 
	 , sum(case when gender = 'М' then 1.0 else 0.0 end)/ count(*) as share_men
from client_info t1
	join city_info t2
    		on t1.id_city = t2.id_city
where age >= 20  and age <= 40
group by name_city;


-- Задание 2.3: Средний возраст неактивных клиентов
SELECT AVG(age) as avg_age
FROM client_info
WHERE id_client NOT IN (SELECT DISTINCT id_client 
          FROM payments 
          WHERE amt_payment > 0
          );


-- Задание 2.4: Первые три платежа по округам
SELECT name_region
	 , time_payment
FROM (SELECT t3.name_region
       	 	  , t1.time_payment
         		  , ROW_NUMBER() OVER (PARTITION BY t3.name_region ORDER BY t1.time_payment) as payment_rank
   	  FROM payments t1
   		 JOIN client_info t2
      			ON t1.id_client = t2.id_client
   		 JOIN city_info t3
      			ON t2.id_city = t3.id_city
	  ) ranked_payments
WHERE payment_rank <= 3
ORDER BY name_region, time_payment;


-- Задание 2.5: Среднее время между платежами
WITH payment_diffs AS (SELECT  t3.name_city
        				        ,  EXTRACT(EPOCH FROM (t1.time_payment::timestamp - 
            LAG(t1.time_payment::timestamp) OVER (PARTITION BY t1.id_client ORDER BY t1.time_payment::timestamp))) as time_diff_seconds
   			        FROM payments t1
    				JOIN client_info t2 ON t1.id_client = t2.id_client
    				JOIN city_info t3 ON t2.id_city = t3.id_city
   			        WHERE t3.name_region IN ('Южный', 'Северный')
       )
SELECT name_city
               , AVG(time_diff_seconds) as avg_seconds
               , FLOOR(AVG(time_diff_seconds) / 86400) || ' дней ' ||
              FLOOR((AVG(time_diff_seconds) % 86400) / 3600) || ' часов ' ||
              FLOOR((AVG(time_diff_seconds) % 3600) / 60) || ' минут ' ||
              FLOOR(AVG(time_diff_seconds) % 60) || ' секунд' as avg_time_between_payments
FROM payment_diffs
WHERE time_diff_seconds IS NOT NULL AND time_diff_seconds > 0
GROUP BY name_city
ORDER BY name_city;

3. Файл create_tables.sql - для тестирования:
sql
-- Пример структуры таблиц для тестирования запросов
CREATE TABLE city_info (
    id_city INT PRIMARY KEY,
    name_city VARCHAR(100),
    name_region VARCHAR(100)
);

CREATE TABLE client_info (
    id_client INT PRIMARY KEY,
    gender VARCHAR(1),
    age INT,
    id_city INT,
    FOREIGN KEY (id_city) REFERENCES city_info(id_city)
);

CREATE TABLE payments (
    id_payment SERIAL PRIMARY KEY,
    id_client INT,
    time_payment VARCHAR(16),
    amt_payment DECIMAL(10,2),
    FOREIGN KEY (id_client) REFERENCES client_info(id_client)
);
