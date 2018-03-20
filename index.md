![LSE](lse-logo.jpg) 
# ST446 Distributed Computing for Big Data 

### Lent Term 2018

### Instructors

* [Milan Vojnovic](http://personal.lse.ac.uk/vojnovic/), [email](mailto:m.vojnovic@lse.ac.uk), Department of Statistics.  *Office hours*: By appointment, COL 5.05

### Teaching Assistant
* [Christine Yuen](http://personal.lse.ac.uk/yuenl/), [email](mailto:L.T.Yuen@lse.ac.uk), Department of Statistics.  *Office hours*: Monday 13:00 - 14:00, COL 5.03 (from week 2)

### Course Information

* Lectures on Mondays 10:00–12:00 in TW2.2.04
* Classes on Thursdays 12:30–14:00 in TW2.4.01

No lectures or classes will take place during School Reading Week 6.

| **Week** | **Topic**                            |
|----------|--------------------------------------|
| 1        | [Introduction](#week-1-introduction) |
| 2        | [Distributed file systems and key-value stores](#week-2-distributed-file-systems-and-key-value-stores)                  |
| 3        | [Distributed computation models](#week-3-distributed-computation-models)    |
| 4        | [Structured queries over large datasets](#week-4-structured-data-management-systems)       |
| 5        | [Graph data processing](#week-5-graph-data-processing) <br> Guest lecturer: [Mark Needham](https://neo4j.com/blog/contributor/mark-needham/), Software Engineer, Neo4j                  |
| 6        | _Reading Week_                       |
| 7        | [Stream data processing](#week-7-stream-data-processing) <br> Guest lecturer: [Eno Thereska](https://enothereska.wordpress.com/), Principal Engineer, Amazon |
| 8        | [Scalable machine learning I](#week-8-scalable-machine-learning-i) <br> Guest lecturer: [Ulrich Paquet](http://ulrichpaquet.com/), Research Scientist, Google DeepMind | 
| 9        | [Scalable machine learning II](#week-9-scalable-machine-learning-ii) <br> Guest lecturer: [Ryota Tomioka](https://www.microsoft.com/en-us/research/people/ryoto/), Researcher, Microsoft Research |
| 10       | [AI applications](#week-10-ai-applications) <br> Guest lecturer: [Marc Cohen](https://about.me/marc1), Software Engineer, Google |
| 11       | [Distributed dataflow graph computations](#week-11-distributed-dataflow-graph-computations)           |




### Course Description

This course covers the main principles and application programming interfaces of distributed systems for storing and processing big data. This includes the principles of distributed file systems, storage systems, and data models that are in common use in modern on-premise data analytics platforms and cloud computing services. The course covers the principles of computing over large datasets in distributed computing systems involving multi-core processors and cluster computing systems. Students will learn how to perform canonical distributed computing tasks in batch, interactive and stream processing settings and how to run scalable machine learning algorithms for regression, classification, clustering and collaborative filtering tasks. 

This course uses a project-based learning approach where students gain hands-on experience in using computing tools, services and writing software code through computer workshop exercises and project assignments. This equips students with key skills and knowledge about modern computation platforms for processing big data. In particular, students gain hands-on experience in working with Apache Hadoop, an open-source software framework for distributed storage and processing of dataset of big data using the MapReduce programming model and other services such as Apache Hive. They also gain hands-on experience in working with  Apache Spark, the fastest-growing general engine for processing big data, used across different industries, and connecting Spark with various data sources and other systems. The students learn how to run big data analytics tasks locally on their laptops as well as on distributed clusters of machines in the cloud. The students work on weekly exercises and project assignments by using GitHub, a popular revision-control and group collaboration tool. Each student develops code for solving one or more computation tasks and uses GitHub for accessing and submitting course materials and assignments.

On the theory side, we introduce the main principles of distributed systems for big data analytics, their design objectives, querying paradigms by using MapReduce and other computation models, general numerical computations using dataflow graphs, and querying data by using SQL-like application programming interfaces. We consider graph processing algorithms for querying graph properties and iterative computations on input graph data. We introduce the principles of stream processing, how to perform computations and execute queries over a sliding-window of input data stream elements. We study the principles of scalable machine learning algorithms, based on parallel implementations of gradient-descent style algorithms for minimizing a loss function, used for training regression and classification models. We also consider distributed MapReduce computations for training clustering models such as k-means and collaborative filtering models based on matrix factorization. We consider numerical computations using dataflow graphs, with a focus on learning deep neural networks for image classification and other classification tasks. Students are encouraged to work on computations and data relevant to their own interests.

On the practical side, we cover a variety of tools that are part of a modern data scientist's toolkit, including distributed computing using Apache Hadoop and MapReduce style processing of big data sets, Apache Spark, application programming interfaces for querying structured and unstructured datasets such as Apache Hive, Spark SQL, and Google BigQuery, stream data processing, and deploying large-scale machine learning models. Students learn how to write programmes to define Spark jobs using the Python API and how to deploy a Spark job in a production environment. Students learn how to connect Spark data structures with a variety of external data sources, including key-value data stores, relational databases, and publish-subscribe messaging systems.

For the final project, students are asked to conduct a big data analytics tasks using the principles and technologies learned in class as well as to learn other related technologies not covered in course in a great length (e.g. working with Apache Cassandra or Microsoft Congitive Toolkit). The project report is typically in the form a Jupyter notebook and a working solution.



### Organization

This course is an introduction to the fundamental concepts of distributed computing for big data for students and assumes no prior knowledge of these concepts.  

The course involves 20 hours of lectures and 15 hours of computer workshops in the LT. 	


### Prerequisites

Some basic prior programming experience is expected. Prior experience with Python programming is desirable; for example, acquired through the compulsory courses of the MSc in Data Science program.


### Software

We use a wide range of tools, including Juypter notebooks, Apache Hadoop, Google Bigtable, Apache Hive, Apache Spark / PySpark (Python API for Spark), SQL APIs for querying datasets, Tensorflow library for dataflow programs, Docker, and various cloud computing services, e.g. provided by the Google Cloud Platform. Lectures and assignments are posted on Github. Students are expected to use Github also to submit problem sets and final exam.

Where appropriate, we use Jupyter notebooks for lab assignments, demonstrations, and the course notes themselves.

### Assessment

Project assignment (80%) and continuous assessment in weeks 4 and 7 (10% each). Students are expected to produce 10 problem sets in the LT. 


### Schedule

---
#### Week 1. Introduction

In the first week, we provide an overview of basic concepts starting with a definition of big data, followed by an overview of structured, semi-structured and unstructured data types, including relational database tables, delimiter-separated file formats such as csv and tsv files, JSON and XML file formats. We then consider main properties of traditional relational database management systems, their support of transactions and ACID properties. This leads us to consider the need for the design of scalable systems, the concepts of horizontal and vertical scaling, and various computer system bottlenecks. We then go on to consider modern big data analytics systems and how they have evolved over time. We introduce various computation paradigms such as batch processing, interactive processing, stream processing, and lambda architecture. We discuss main developments such as MapReduce computation model and noSQL databases. The rest of the lecture discusses various computation tasks that led to the development of modern big data analytics systems, which are studied throughout the course. 


*Readings*:
* Vogels, W., [A Decade of Dynamo](http://www.allthingsdistributed.com/2017/10/a-decade-of-dynamo.html), Blog, All Things Distributed, 02 October 2017
* Hamilton, J., [One Size Doesn't Fit All](http://perspectives.mvdirona.com/2009/11/one-size-does-not-fit-all/), Blog, Perspectives, 2012


*Lab*: **Getting started** 
* Command line interface and commands
* Cluster and bucket creation in a cloud
* Submitting a simple "Hello World" job to a cluster
* Running a Jupyter notebook on a cluster
* Use of Docker containers

---
#### Week 2. Distributed file systems and key-value stores

In this week, we first consider the main design principles of distributed file systems, explaining the original Google File System and its refinements, as well as other distributed file systems such as Hadoop Distributed File System (HDFS). We then consider the main design principles of distributed key-value stores such as Amazon Dynamo and columnar data storage systems such as BigTable and Apache Cassandra. 

*Readings*:
* White, T., [Hadoop: The Definitive Guide](https://www.amazon.co.uk/Hadoop-Definitive-Guide-Tom-White/dp/1491901632/ref=sr_1_1?ie=UTF8&qid=1514806006&sr=8-1&keywords=hadoop+the+definitive+guide), O'Reilly, 4th Edition, 2015
* Carpenter, J. and Hewitt, E., [Cassandra: The Definitive Guide](https://www.amazon.co.uk/Cassandra-Definitive-Guide-Jeff-Carpenter/dp/1491933666/ref=sr_1_1?ie=UTF8&qid=1514834275&sr=8-1&keywords=cassandra+definitive+guide), 2nd Edition, O'Reilly, 2016 
* Ghemawat, S., Gobioff, H. and Leung S.-T., [The Google file system](https://research.google.com/archive/gfs.html), SOSP 2003
* Shvachko, K. et al, [The Hadoop Distributed File System](http://storageconference.us/2010/Papers/MSST/Shvachko.pdf), IEEE MSST 2010; see also [html](http://www.aosabook.org/en/hdfs.html)
* DeCandia, G. et al, [Dynamo: Amazon’s Highly Available Key-value Store](http://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf), SOSP 2007
* Chang, F. et al, [Bigtable: A distributed storage system for structured data](https://static.googleusercontent.com/media/research.google.com/en//archive/bigtable-osdi06.pdf), OSDI 2006


*Further Readings*:
* [GFS: Evolution on Fast-Forward](http://queue.acm.org/detail.cfm?id=1594206), ACM Queue, Vol 7, No 7, August, 2009
* Apache Hadoop docs: [HDFS Architecture](http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html) 
* Vogels, W., [Amazon's Dynamo](http://www.allthingsdistributed.com/2007/10/amazons_dynamo.html), Blog, 2007
* Nishtala, R. et al, [Scaling Memcache at Facebook](https://www.usenix.org/node/172909), NSDI 2013
* Fitzpatrick, P., [Distributed Caching with Memcached](http://www.linuxjournal.com/article/7451), 2004
* Lakshman, A. and Malik, K., [A Decentralized Structured Storage System](http://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf), LADIS 2009
* Calder, B. et al, [Windows Azure Storage: A Highly Available Cloud Storage Service with Strong Consistency](http://sigops.org/sosp/sosp11/current/2011-Cascais/printable/11-calder.pdf), SOSP 2011
* Huang, C. et al, [Erasure Coding in Windows Azure Storage](https://www.usenix.org/node/168894), USENIX 2012

*Lab*: **System installation and API practice**
* Go through Hadoop installation
* Basic file manipulation commands working with HDFS
* Reading and writing data from BigTable

---
#### Week 3. Distributed computation models

In this lecture we explain the basic principles of distributed computation models that are in common use in the context of big data analytics systems. We start with explaining MapReduce computation model that is in widespread use for distributed processing of large datasets. We then move on to consider Pregel computation model, developed for iterative computations such as computing PageRank vector for a large-scale input graph. Finally, we consider the concept of a Resilient Distributed Dataset (RDD), a distributed memory abstraction that lets programmers perform in-memory computations on large clusters in a fault-tolerant manner. This involves to consider the types of operations performed on RDDs and their execution on a distributed cluster of machines. 

*Readings*:
* Karau, H., Konwinski, A., Wendell, P. and Zaharia, M., [Learning Spark: Lightining-fast Data Analysis](https://www.amazon.co.uk/Learning-Spark-Lightning-Fast-Data-Analysis/dp/1449358624), O'Reilly, 2015
* Karau, H. and Warren, R., [High Performance Spark: Best Practices for Scaling & Optimizing Apache Spark](https://www.amazon.co.uk/High-Performance-Spark-Practices-Optimizing/dp/1491943203), O'Reilly, 2017
* Drabas, T. and Lee D., [Learning PySpark](https://www.amazon.co.uk/Learning-PySpark-Tomasz-Drabas/dp/1786463709/ref=sr_1_1?ie=UTF8&qid=1515747362&sr=8-1&keywords=learning+pyspark), Packt, 2016
* Dean, J. and Ghemawat, S., [Mapreduce: Simiplified Data Processing on Large Clusters](https://courses.cs.washington.edu/courses/cse547/17sp/content/Downloads/p107-dean.pdf), Communications of the ACM, Vol 51, No 1, January 2008; original [OSDI 2004 paper](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)
* Zaharia, M. et al, [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](http://people.csail.mit.edu/matei/papers/2012/nsdi_spark.pdf), NSDI 2012

*Further Readings*:
* Apache Hadoop [documentation](http://hadoop.apache.org/docs/r3.0.0/)
* Apache Hadoop documentation: [MapReduce Tutorial](http://hadoop.apache.org/docs/r3.0.0/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)
* Malewicz, G. et al, [Pregel: A System for Large-Scale Graph Processing](https://www.cs.cmu.edu/~pavlo/courses/fall2013/static/papers/p135-malewicz.pdf), SIGMOD 2010
* Spark programming [guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)   
* Chambers, B. and Zaharia, M., Spark: The Definitive Guide, databricks, 2017
* Spark documentation: [PySpark package](http://spark.apache.org/docs/2.1.0/api/python/pyspark.html)

*Lab*: **MapReduce and resilient distributed datasets**
* Run a MapReduce job on Hadoop for word count on dblp data
* Hands-on experience with running operations on resilient distributed datasets in PySpark, such as map, flatMap, filter, distinct, sample, leftOuter and repartition, and actions such as take, collect, reduce, count, saveAsTextFile and foreach
* Run the word count example using resilient distributed datasets in PySpark

---
#### Week 4. Structured queries over large datasets

In this week we consider systems for big data analytics using structured query languages. We start with considering the main architectural principles of Apache Hive, a data warehouse solution running on top of Apache Hadoop. We consider data types and query language used by Hive. We then consider the main design principles of Dremel (BigQuery) and Spark SQL for querying data using structured query languages. 

*Readings*:
* White, T., [Hadoop: The Definitive Guide](https://www.amazon.co.uk/Hadoop-Definitive-Guide-Tom-White/dp/1491901632/ref=sr_1_1?ie=UTF8&qid=1514806006&sr=8-1&keywords=hadoop+the+definitive+guide), O'Reilly, 4th Edition, 2015
* Karau, H., Konwinski, A., Wendell, P. and Zaharia, M., [Learning Spark: Lightining-fast Data Analysis](https://www.amazon.co.uk/Learning-Spark-Lightning-Fast-Data-Analysis/dp/1449358624), Chapter 9 Spark SQL, O'Reilly, 2015
* Karau, H. and Warren, R., [High Performance Spark: Best Practices for Scaling & Optimizing Apache Spark](https://www.amazon.co.uk/High-Performance-Spark-Practices-Optimizing/dp/1491943203/ref=pd_lpo_sbs_14_t_1?_encoding=UTF8&psc=1&refRID=6QE38GNZN2YBJ5SS99FP), O'Reilly, 2017
* Drabas, T. and Lee D., [Learning PySpark](https://www.amazon.co.uk/Learning-PySpark-Tomasz-Drabas/dp/1786463709), Chapter 3 DataFrames, Packt, 2016
* Rutherglen, J., Wampler, D., Capriolo, E., [Programming Hive](https://www.safaribooksonline.com/library/view/programming-hive/9781449326944/), 2nd Edition, O'Reilly, 2017
* Prokopp, C., [The Free Hive Book](https://github.com/Prokopp/the-free-hive-book)

*Further Readings*:
* Apache Hive [Tutorial](https://cwiki.apache.org/confluence/display/Hive/Tutorial)
* Apache Hive [Language Manual](https://cwiki.apache.org/confluence/display/Hive/LanguageManual)
* Thusoo, A. et al, [Hive-A Petabyte Scale Data Warhouse Using Hadoop](http://infolab.stanford.edu/~ragho/hive-icde2010.pdf), ICDE 2010
* Spark SQL programming guide: [Spark SQL, DataFrames and Datasets Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html), Apache Spark 2.2.0, 2017
* Armbrust, M., et al, [Spark SQL: Relational Data Processing in Spark](https://amplab.cs.berkeley.edu/wp-content/uploads/2015/03/SparkSQLSigmod2015.pdf), ACM SIGMOD 2015
* Armbrust, M., [Dataframes: Simple and Fast Analysis of Structured Data](http://go.databricks.com/databricks-webinar-spark-dataframes-simple-and-fast-analysis-of-structured-data-0), Webinar, 2017
* Big Query SQL Reference: [Standard](https://cloud.google.com/bigquery/docs/reference/standard-sql/) and [Legacy](https://cloud.google.com/bigquery/docs/reference/legacy-sql)

*Lab*: **Hive and Spark SQL queries**
* Run Hive queries, basic standard SQL and Hive specific queries such as TRANSFORM, and MAP and REDUCE, queries
* Loading data from sources, including JSON, XML, weblogs using regular expressions 
* Run queries in Spark SQL using dataframe API and Spark Session SQL API
* Data management using BigQuery via web interface and connector with Python

Note: assignment for grading to be given in this week

---
#### Week 5. Graph data processing

In this week we consider principles and systems for scalable processing of large-scale graph data. This includes queries such as evaluating node centralities (e.g. degree centrality), graph traversal or motif queries for finding structures in graph data (e.g. identifying friends-of-friends of a person who were born in London), and iterative algorithms on graph input data (e.g. computing PageRank node centralities). We discuss different data models for representation of graph data such as [RDF](http://www.w3.org/TR/rdf-sparql-query/), as well as query languages, including [SPARQL](http://www.w3.org/TR/rdf-sparql-query/), [Gremlin](http://tinkerpop.apache.org/), [Cypher](https://neo4j.com/developer/cypher/) and [openCyper](http://www.opencypher.org/) that used in graph databases. We introduce the Bulk Synchronous Parallel computation model that underlies the design of modern computation platforms for iterative computing on input graph data. 


*Readings*:
* Drabas, T. and Karau, H., _Learning PySpark_, Chapter 7: GraphFrames, Packt, 2017
* Spark [GraphX: programming guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
* Gonzalez, J. E. et al, [GraphX: Graph Processing in Distributed Dataflow Framework](https://www.usenix.org/node/186217), OSDI 2014

*Further Resources*:

* Malewicz, G. et al, [Pregel: A System for Large-Scale Graph Processing](https://kowshik.github.io/JPregel/pregel_paper.pdf), ACM SIGMOD 2010; open source cousin: [Apache Giraph](http://giraph.apache.org/)
* Low, Y. et al, [Distributed GraphLab: A Framework for Machine Learning and Data Mining in the Cloud](http://vldb.org/pvldb/vol5/p716_yuchenglow_vldb2012.pdf), VLDB 2012
* Valiant, L. G., [A Bridging Model for Parallel Computation](http://web.mit.edu/6.976/www/handout/valiant2.pdf), Communications of the ACM, Vol 3, No 8, August 1990.

*Lab*: **Analysis of StackExchange user-posts-topics relations**
* Importing StackExchange relations into graphframes
* Computing degree and PageRank node centralities
* Answering graph motif queries
* Breadth-first search and connected components


---
#### Week 6. Reading week

---
#### Week 7. Stream data processing

In this week we consider basic concepts of data stream processing systems. We explain various global aggregators, cumulative, and sliding-window stream data processing tasks. We introduce the concept of publish-subscribe systems, taking Apache Kafka as an example. We discuss the importance of fault tolerance in stream processing systems and discuss fault tolerance models such as _execute exactly-once_. In this context, we discuss the guarantees provided by Zookeper, an open source server which enables highly reliable distributed coordination.  

*Readings*:
* Spark [streaming programming guide](http://spark.apache.org/docs/latest/streaming-programming-guide.html)
* Spark [structured streaming programming guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)


*Further Resources*:
* Zaharia, M. et al, [Discretized Streams: Fault-Tolerant Streaming Computation at Scale](http://people.csail.mit.edu/matei/papers/2013/sosp_spark_streaming.pdf), SOSP 2013
* Apache [Kafka documentation](https://kafka.apache.org/documentation/)

*Lab*: **Twitter feed processing**
* Import Twitter feed into Kafka as a topic
* Integrate Spark with Kafka
* Track heavy-hitter topics
* Visualize topic trends

---
#### Week 8. Scalable machine learning I

In this week we introduce the basic concepts of scalable distributed machine learning algorithms for regression and classification tasks. We discuss batch optimization methods for model parameter estimation using iterative methods such as gradient descent, batch gradient descent, stochastic gradient descent, and quasi-Newton methods such as [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) and [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS). 

*Readings*:
* Bottou, L. and Le Cun, Y., [Large Scale Online Learning](http://papers.nips.cc/paper/2365-large-scale-online-learning), NIPS 2003
* Drabas, T. and Lee, D.  _Learning PySpark_, Chapter 5: Intoducing MLib, Packt, 2017
* Spark programming guide [Machine Learning Library (MLlib) Guide](https://spark.apache.org/docs/2.2.0/ml-guide.html) 

*Further Resources*:
* Li, M., [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf), OSDI 2014
* [Numerical optimization: understanding L-BFGS](http://aria42.com/blog/2014/12/understanding-lbfgs), blog, December 2, 2014
* Machine Learning [Glossary](https://developers.google.com/machine-learning/glossary)

*Lab*: **Logistic regression using Spark MLlib**
* Training a logistic regression model using batch gradient descent in Spark MLlib
* Comparison of stochastic gradient descent and L-BFGS methods

Note: assignment for grading to be given in this week

---
#### Week 9. Scalable machine learning II

In this week we continue by considering distributed machine learning algorithms for learning deep neural networks, and consider distributed algorithms for other machine learning problems such as collaborative filtering for recommendation systems and topic model for text data analysis. 

for clustering and collaborative filtering tasks. We discuss a MapReduce algorithm for k-means clustering problem, as well as an iterative algorithm for collaborative filtering tasks. We consider Spark API approaches provided by MLib and ML packages. In the latter context, we introduce the concept of a pipeline that consists of a dataflow passing through transformer and estimator operators. 


*Readings*:
* Drabas, T. and Lee, D.  _Learning PySpark_, Chapter 5: Intoducing MLib and Chapter 6: Introducting the ML Package, Packt, 2017
* Spark programming guide [Machine Learning Library (MLlib) Guide](https://spark.apache.org/docs/2.2.0/ml-guide.html) 

*Further Resources*:
* Zadeh, R. et al, [Matrix Computations and Optimizations in Spark](http://www.kdd.org/kdd2016/subtopic/view/matrix-computations-and-optimization-in-apache-spark), KDD 2016
* Koren, Y., Bell, R. and Volinsky, C., [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf), Computer, Vol 42, No 8, 2009
* Deerwester, S., Dumais, S. T., and Harshman, R., [Indexing by Latent Semantic Analysis](http://www.psychology.uwo.ca/faculty/harshman/latentsa.pdf), Journal of the American Society for Information Science, Vol 41, No 6, 391-407, 1990
* Blei, D. M., Ng, A. Y., and Jordan, M., I., [Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf), JMLR 2003 
* Hoffman, M., Bach, F. R., and Blei, D. M., [Online Learning for Latent Dirichlet Allocation](https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation), NIPS 2010



*Lab*: **Collaborative filtering and topic modelling using Spark MLlib**
* Movielens movie recommendation problem using Alternating Least Squares
* Latent semantic indexing using singular value decomposition
* Topic modelling using Latent Dirichlet Allocation

---
#### Week 10. Cloud computing services

Guest lecture: "Data Science in the Cloud," Marc Cohen, Software Engineer, Google. 

The lecture will provide an overview of cloud services provided by Google including Compute Engine, Cloud ML Engine, Vision API, Translation API, Video intelligence API, and managed TensorFlow at scale. 

*Lab*: **Introduction to deep learning using TensorFlow**
* TenorFlow and Deep Learning, without a PhD [Google codelab](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)

---
#### Week 11. Distributed dataflow graph computations

In our last lecture, we introduce basic concepts of distributed computing using dataflow graphs. In such settings, the user defines a dataflow graph where nodes of the graph represent operations (e.g. a matrix multiplication, a non-linear function) and edges represent flow of data between operations. We will primarily focus on one system that uses dataflow graph computations, namely, TensorFlow, used for learning and inference for deep neural networks. We will explain the key system architectural concepts that underlie the design of TensorFlow as well as the application programming interface. 

*Readings*:
* Abadi, M. et al, [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi), OSDI 2016
* DistBelief: Dean, J., [Large Scale Distributed Deep Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40565.pdf), NIPS 2012
* TensorFlow [API docs](https://www.tensorflow.org/api_docs/)


*Further Resources*:
* Tensorflow [gitHub page](https://github.com/tensorflow/tensorflow)
* Abadi et al, [A computational model for TensorFlow: an introduction](https://dl.acm.org/citation.cfm?doid=3088525.3088527), MAPL 2017
* Dean, J., [Large-Scale Deep Learning with Tensorflow](https://www.matroid.com/scaledml/slides/jeff.pdf), ScaledML 2016

*Lab*: **Distributed TensorFlow**
* Google Cloud Platform: [Running Distributed TensorFlow on Compute Engine](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)
* https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example
* [Using Distributed TensorFlow with Cloud ML Engine and Cloud Datalab](https://cloud.google.com/ml-engine/docs/distributed-tensorflow-mnist-cloud-datalab)
