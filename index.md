![LSE](images/lse-logo.jpg) 
# ST446 Distributed Computing for Big Data 

### Lent Term 2018

### Instructors

* [Milan Vojnovic](mailto:m.vojnovic@lse.ac.uk), Department of Statistics.  *Office hours*: By appointment, COL 5.05

### Teaching Assistant
* [Christine Yuen](mailto:L.T.Yuen@lse.ac.uk), Department of Statistics.  *Office hours*: Thursday 11:00 - 12:00, COL 5.03

### Course Information

* Lectures on Mondays 10:00–12:00 in TW2.2.04
* Classes on Thursdays 12:30–14:00 in TW2.4.01

No lectures or classes will take place during School Reading Week 6.

| **Week** | **Topic**                            |
|----------|--------------------------------------|
| 1        | [Introduction to basic concepts and system architectures](#week-1-introduction-to-basic-concepts-and-system-architectures) |
| 2        | [Databases and data storage systems](#week-2-databases-and-data-storage-systems)                  |
| 3        | [Querying unstructured datasets](#week-3-querying-unstructured-datasets)    |
| 4        | [Querying structured datasets](#week-4-querying-structured-datasets)       |
| 5        | [Graph data processing](#week-5-graph-data-processing)                  |
| 6        | _Reading Week_                       |
| 7        | [Stream data processing](#week-7-stream-data-processing) Guest lecturer: Eno Thereska, *Amazon* |
| 8        | [Scalable machine learning I](#week-8-scalable-machine-learning-i) |
| 9        | [Scalable machine learning II](#week-9-scalable-machine-learning-ii) Guest lecturer: Ryoto Tomioka, *Microsoft Research* |
| 10       | [Numerical computations using data flow graphs](#week-10-numerical-computations-using-data-flow-graphs) Guest lecturer: Marc Cohen, *Google* |
| 11       | [Deployment of computation jobs in production](#week-11-deployment-of-computation-jobs-in-production)           |




### Course Description

This course will cover the principles of distributed systems for storing and processing big data. This will include the principles of storage systems, databases and data models that are in common use by on-premise data analytics platforms and cloud computing services. The course will cover the principles of computing over large datasets in distributed computing systems involving multi-core processors and cluster computing systems. Students will learn how to perform canonical distributed computing tasks in batch, streaming and graph processing computation models and how to run scalable machine learning algorithms for regression, classification, clustering and collaborative filtering tasks. This course uses a project-based learning approach where students gain hands-on experience with writing and running computer programmes through computer workshop exercises and project assignments. This will equip students with key skills and knowledge about modern computation platforms for processing big data. In particular, students will get hands-on experience with using Apache Spark, the fastest-growing general engine for processing big data that is used across different industries, and connecting Spark programmes with various databases and other systems. The students will work on weekly exercises and project assignments by using revision-control and group collaboration tools such as GitHub. Each student will develop code for solving one or more computation tasks on an input dataset, and will use GitHub for accessing and submitting course materials and assignments.

On the theory side, we will introduce principles of distributed databases, their design objectives, querying paradigms by using MapReduce style of computations, general numerical computations using dataflow graphs, and querying using SQL application programming interfaces. We will consider graph processing algorithms, for querying graph properties and iterative computations using input graph data. We will also introduce the principles of stream processing, how to perform computations and execute queries over a sliding-window of input data stream elements. We will study the principles of scalable machine learning algorithms that are based on parallel implementations of gradient descent style algorithms for minimizing a loss function, used for training regression and classification models. We will also consider distributed MapReduce based computations for training clustering models such as k-means and collaborative filtering models based on matrix factorization. We will consider numerical computations using dataflow graphs, with a focus on the use case of learning a deep neural network for image classification and other classification tasks. Students will be encouraged to work with computations and data relevant to their own interests.

On the practical side, we will cover a variety of tools that are part of a modern data scientist's toolkit, including distributed computing using Apache Spark, Mapreduce style processing of big data sets, application programming interfaces for querying structured and unstructured datasets, stream data processing, and deploying large-scale machine learning models. You will learn how to write programmes to define Spark jobs using the Python API and how to deploy a Spark job in a production environment. You will learn how to connect Spark data structures with a variety of external data sources, including key-value databases, relational databases, and publish-subscribe messaging systems.

For the final project, we will ask you to develop and run a distributed computation for a given dataset, which you will be expected to implement in a PySpark Jupyter notebook.



### Organization

This course is an introduction to the fundamental concepts of distributed computing for big data for students and assumes no prior knowledge of these concepts.  

The course will involve 20 hours of lectures and 15 hours of computer workshops in the LT. 	


### Prerequisites

Some basic prior programming experience is expected. Prior experience with Python programming is desirable; for example, acquired through the compulsory courses of the MSc in Data Science program.


### Software

We will use some tools, notably Apache Spark general engine for computing over large distributed datasets, PySpark (Python API for Spark), SQL APIs for querying datasets, and Tensorflow library for dataflow programmes. Lectures and assignments will be posted on Github, Students are expected to use Github also to submit problem sets and final exam.

Where appropriate, we will use Jupyter notebooks for lab assignments, demonstrations, and the course notes themselves.

### Assessment

Project assignment (80%) and continuous assessment in weeks 4 and 7 (10% each). Students will be expected to produce 10 problem sets in the LT. 


### Schedule

---
#### Week 1. Introduction to basic concepts and system architectures

In the first week, we will introduce the basic concepts and system architectures for big data processing. We will introduce the basic computing paradigms of batch, streaming, imperative, declarative, graph and machine learning data processing. We will discuss the main architectures of data storage systems based on key-value stores and other data models. We will discss the main design goals of such systems such as consistency, optimization for fast and reliable read or writes. We will then introduce the basic concepts of multi-node computing, such as cluster computing systems consisting of multiple machines, multi-core processors, distributed file systems, partitions of large data files into chunks or extents, distributed computing using master and worker nodes, resource allocation through job scheduling using resource managament systems such as YARN and Mesos.


*Readings*:
* Zaharia, M. et al, [Apache Spark: A Unified Engine for Big Data Processing](https://cacm.acm.org/magazines/2016/11/209116-apache-spark/fulltext), Communications of the ACM, Vol 59, No 11, November 2016
* Drabas, T. and Lee, D., _Learning PySpark_, Chapter 1: Understanding Spark, Packt, 2017


*Further Resources*:
* Manual [Installing Spark](https://www.packtpub.com/sites/default/files/downloads/InstallingSpark.pdf)
* Mastering Apache Spark 2, [Task Scheduler](https://www.tecmint.com/command-line-tools-to-monitor-linux-performance/) 
* DAG scheduler http://bit.ly/29WTiKB
* Matt Turcks's [Big Data Landscape](http://mattturck.com/wp-content/uploads/2016/03/Big-Data-Landscape-2016-v18-FINAL.png) 


*Lab*: **Hands-on system administration tools** 
* Use of basic linux/Mac OS/Windows command line utilities
* Getting to know your cluster system, processors and machines
* Use of cloud computing services: AWS example
* Use of docker images

---
#### Week 2. Databases and data storage systems

In this week we will introduce different data models, datasets, databases and data storage paradigms used for distributed computing for big data. We will discuss key-value databases such as Cassandra and more complex relational database models such as Hive. We will discuss different data formats for storing data including csv, tsv, JSON, XML, Parquet, Hive tables, RDF, and Azure blobs. We will introduce the basic data structures used in Spark, including Resilient Distributed Dataset (RDD) and DataFrame. We will discuss the design objectives of various large-scale distributed storage systems such as consistency and fast reads or writes. 


*Readings*:
* Hamilton, J., [One Size Doesn't Fit All](http://perspectives.mvdirona.com/2009/11/one-size-does-not-fit-all/), Blog, 2012
* Drabas, T. and Lee, D., _Learning PySpark_, Chapter 2: Resilient Distributed Datasets, Chapter 3: DataFrames, Packt, 2017
* Apache [Cassandra](http://cassandra.apache.org/) [Documentation](http://cassandra.apache.org/doc/latest/)
* Apache Hive [Tutorial](https://cwiki.apache.org/confluence/display/Hive/Tutorial)

*Further Resources*:
* Vogels, W., [Amazon's Dynamo](http://www.allthingsdistributed.com/2007/10/amazons_dynamo.html), Blog, 2007
* Fitzpatrick, B., [Distributed Cashing with Memcashed](http://www.linuxjournal.com/article/7451), Linux Journal, 2004
* Nishtala, R. et al, [Scaling Memcashe at Facebook](https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala), NSDI 2013
* Zhou, J. et al, [SCOPE: parallel databases meet MapReduce](http://www.cs.columbia.edu/~jrzhou/pub/Scope-VLDBJ.pdf), VLDB journal, 2012
* Melnik S. et al, [Dremel: Interactive Analysis of Web-Scale Datasets](https://research.google.com/pubs/pub36632.html), VLDB 2010
* Google, [An Inside Look at Google BigQuery](https://cloud.google.com/files/BigQueryTechnicalWP.pdf), White Paper, 2012
* Amazon Web Services [Documentation](https://aws.amazon.com/documentation/), [White Papers](https://aws.amazon.com/whitepapers/)
* Google Cloud [Documentation](https://cloud.google.com/docs/)
* Microsoft Azure [Storage Documentation](https://docs.microsoft.com/en-us/azure/storage/), [SOSP 2011 paper & slides](https://blogs.msdn.microsoft.com/windowsazurestorage/2011/11/20/sosp-paper-windows-azure-storage-a-highly-available-cloud-storage-service-with-strong-consistency/), [Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)

*Lab*: **Import data to RDD and DataFrame**
* Create an RDD from a data collection and a file
* Transform an RDD to a dataframe and vice-versa
* Import data from JSON and XML files
* Import data from a key-value database
* Import data from a postgreSQL database

---
#### Week 3. Querying unstructured datasets

In this week we will study how to query large unstructured datasets. We will introduce the parallel computing paradigm of MapReduce. We will discuss how to manage and query datasets using Spark RDD. We will learn how to create an RDD, use transformations such as map, flatMap, filter, distinct, sample, leftOuterJoin, repartition as well as actions such as take, collect, reduce, count saveAsTextFile, and foreach. We will introduce the concept of lambda expressions and how to use regular expressions.


*Readings*:
* Dean, J. and Ghemawat, S., [Mapreduce: Simiplified Data Processing on Large Clusters](https://courses.cs.washington.edu/courses/cse547/17sp/content/Downloads/p107-dean.pdf), Communications of the ACM, Vol 51, No 1, January 2008, [OSDI 2004 version](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)
* Drabas, T. and Lee, D., _Learning PySpark_, Chapter 2: Resilient Distributed Datasets, Packt, 2017
* Karau, H. et al, _Learning Spark:Lightning-Fast Data Analysis_, O'Reilly 2015 
* Karau, H and Warren R., _High Performance Spark: Best Practices for Scaling & Optimizing Apache Spark_, O'Reilly 2017
* Laskowski, J., [_Mastering Apache Spark 2_](https://jaceklaskowski.gitbooks.io/mastering-apache-spark/)
* Spark Programming Guide http://spark.apache.org/docs/latests/programming-guide.html#rdd-operations
 

*Further Resources*:
* Zaharia, M. et al, [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/node/162809), NSDI 2012
* High Performance Spark, Chapter 5, Effective transformations, section Narrow vs. Wide Transformations, https://smile.amazon.com/High-Performance-Spark-Practices-Optimizing/dp/1491943203
* Regex patterns https://www.packtpub.com/application-development/master-python-regular-expressions

*Lab*: **Using RDDs and MapReduce tasks**
* Use of lambda expressions
* Use of transformations such as map, filter, flatMap, distinct, sample, leftOuterJoin, and repartition
* Use of actions such as take, collect, reduce, count, saveAsTextFile and foreach

---
#### Week 4. Querying structured datasets

In this week we will consider how to query datasets that have a schema. We will introduce the concept of a dataframe and learn  how to query data by using dataframe query API and how to execute SQL queries. We will discuss computational complexity of different standard queries and query optimization techniques.  We will consider how to compute fast approximate query answers by using sampling techniques such as reservoir sampling, and data summarizations or sketches such as hyperloglog sketch for approximating the number of distinct elements in a multiset and count-min for frequency estimation.


*Readings*:
* Drabas, T. and Lee, D., _Learning PySpark_, Chapter 3: DataFrames, Packt, 2017
* Spark [SQL, DataFrames, and Datasets Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)

*Further Resources*:
* Armbrust et al, [Spark SQL: Relational Data Processing in Spark](https://people.csail.mit.edu/matei/papers/2015/sigmod_spark_sql.pdf), ACM SIGMOD 2015
* Zhou, J. et al, [SCOPE: parallel databases meet MapReduce](http://www.cs.columbia.edu/~jrzhou/pub/Scope-VLDBJ.pdf), VLDB journal, 2012
* Melnik S. et al, [Dremel: Interactive Analysis of Web-Scale Datasets](https://research.google.com/pubs/pub36632.html), VLDB 2010
* Cormode, G., [Data Sketching](https://cacm.acm.org/magazines/2017/9/220427-data-sketching/fulltext), Communications of the ACM, Vol 60, No 9, September 2017

*Lab*: **SQL queries on table data**
* Creating a datframe, querying with the dataframe API, querying with SQL
* Computing approximate query answers
* GitHub and StackExchange data analysis using Google Big Query


---
#### Week 5. Graph data processing

In this week we will consider principles and systems for scalable processing of large-scale graph data. This include queries such as evaluating node centralities (e.g. degree centrality), graph traversal or motif queries for finding structural in graph data (e.g. identifying friends-of-friends of a person who were born in London), and iterative algorithms on graph input data (e.g. computing PageRank node centralities). We will discuss different data models for representation of graph data such as [RDF](http://www.w3.org/TR/rdf-sparql-query/), as well as query languages, including [SPARQL](http://www.w3.org/TR/rdf-sparql-query/), [Gremlin](http://tinkerpop.apache.org/), [Cypher](https://neo4j.com/developer/cypher/) and [openCyper](http://www.opencypher.org/) that used in graph databases. We will introduce the bulk synchronous parallel computation model that underlies the design of modern computation platforms for iterative computing on input graph data. 


*Readings*:
* Drabas, T. and Karau, H., _Learning PySpark_, Chapter 7: GraphFrames, Packt, 2017
* Spark [GraphX: programming guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
(http://giraph.apache.org/)
* Gonzalez, J. E. et al, [GraphX: Graph Processing in Distributed Dataflow Framework](https://www.usenix.org/node/186217), OSDI 2014

*Further Resources*:

* Malewicz, G. et al, [Pregel: A System for Large-Scale Graph Processing](https://kowshik.github.io/JPregel/pregel_paper.pdf), ACM SIGMOD 2010; open source cousin: [Apache Giraph]
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

In this week we will consider the basic concepts of data stream processing systems. We will explain various global aggregators, cumulative, and sliding-window stream data processing tasks. We will introduce the concept of publish-subscribe systems and use Apache Kafka as an example. We will discuss the importance of fault tolerance in stream processing systems and discuss fault tolerance models such as execute exactly-once. In this context, we will discuss the guarantees provided by Zookeper, an open source server which enables highly reliable distributed coordination.  

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

In this week we will introduce the basic concepts of distributed machine learning algorithms for regression and classification tasks. We will discuss batch optimization methods for model parameter estimation by using gradient descent methods and its variations such as [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) and [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS). We will also cover online optimisation methods such as stochastic gradient descent (SGD), parallel SGD and mini-batch SGD methods. We will discuss as model and data paralellisation models. 

*Readings*:
* Bottou, L. and Le Cun, Y., [Large Scale Online Learning](http://papers.nips.cc/paper/2365-large-scale-online-learning), NIPS 2003
* Drabas, T. and Lee, D.  _Learning PySpark_, Chapter 5: Intoducing MLib, Packt, 2017
* Spark programming guide [MLib: RDD-based API](https://spark.apache.org/docs/latest/mllib-guide.html) 

*Further Resources*:
* Li, M., [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf), OSDI 2014
* [Numerical optimization: understanding L-BFGS](http://aria42.com/blog/2014/12/understanding-lbfgs), blog, December 2, 2014
* Machine Learning [Glossary](https://developers.google.com/machine-learning/glossary)

*Lab*: **Churn prediction using MLib package**
* Import the Orange Telecoms Churn dataset
* Train a logistic regression model, compute predictions
* Train a decision tree model, compute predictions
* Evaluate and compare the two models


---
#### Week 9. Scalable machine learning II

In this week we will continue by considering distributed machine learning algorithms for clustering and collaborative filtering tasks. We will discuss a Mapreduce algorithm for k-means clustering problem, as well as an iterative algorithm for a collaborative filtering problem. We will consider Spark API approaches provided by MLib and ML packages. For the latter, we will introduce the concept of a pipeline that consists of a dataflow passing through transformer and estimator operators. 


*Readings*:
* Murphy, K. P., _Machine Learning: A Probabilistic Perspective_, k-means, Section 11.4.2.5, The MIT Press, 2012  
* Drabas, T. and Lee, D.  _Learning PySpark_, Chapter 5: Intoducing MLib and Chapter 6: Introducting the ML Package, Packt, 2017
* Spark programming guide [MLib: RDD-based API](https://spark.apache.org/docs/latest/mllib-guide.html) 

*Further Resources*:
* Moore, A., [K-means and Hierarchical Clustering Tutorial](https://www.autonlab.org/tutorials/kmeans.html)
* Wang, Q., [Spark machine learning pipeline by example](https://community.hortonworks.com/articles/53903/spark-machine-learning-pipeline-by-example.html), August 31, 2016
* Zadeh, R. et al, [Matrix Computations and Optimizations in Spark](http://www.kdd.org/kdd2016/subtopic/view/matrix-computations-and-optimization-in-apache-spark), KDD 2016

*Lab*: **Clustering and movies recommendation**
* k-means clustering using a Mapreduce algorithm
* Movie recommendations using MovieLens data and training a collaborative filtering model using Alternating Least Square (ALS) algorithm

---
#### Week 10. Numerical computations using data flow graphs

We will introduce the basic concepts of performing numerical computations using data flow graphs. In such settings, the graph nodes represent mathematical operations, while the graph edges represent the multidimensional data arrays that flow between them. We will explain the architecture of Tensorflow, an open source library for numerical computations using data flow graphs. We will go over the the use case of learning a deep neural network, taking the basic architecture of a feedforward deep neural network. 

*Readings*:
* TensorFlow [API docs](https://www.tensorflow.org/api_docs/)
* Abadi et al, [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi), OSDI 2016
* Drabas, T. and Lee, D., _Learning PySpark_, Chapter 8: TensorFrames, Packt, 2017
* TensorFrames [gitHub page](https://github.com/databricks/tensorframes)

*Further Resources*:
* Tensorflow [gitHub page](https://github.com/tensorflow/tensorflow)
* Abadi et al, [A computational model for TensorFlow: an introduction](https://dl.acm.org/citation.cfm?doid=3088525.3088527), MAPL 2017
* Dean, J., [Large-Scale Deep Learning with Tensorflow](https://www.matroid.com/scaledml/slides/jeff.pdf), ScaledML 2016
* Yu, D. et al, [An Introduction to Computational Networks and the Computational Network Toolkit](https://www.microsoft.com/en-us/research/publication/an-introduction-to-computational-networks-and-the-computational-network-toolkit/), Microsoft Research Technical Report, 2014 


*Lab*: **Deep neural network learning**
* Import a dataset of labeled images
* Specify a feedforward deep neural network model
* Train the model
* Evaluate the classification accuarcy of the trained model

---
#### Week 11. Deployment of computation jobs in production

In the last week, we will discuss how to deploy large-scale computations in a production cluster system. This will cover setting up a cluster system, running jobs over varied number of machines in the cluster, and tracking their progress. We will consider simple Mapreduce jobs as well as machine learning algorithms for prediction tasks on a large-scale data. 

*Readings*:
* Drabas, T. and Lee, D., _Learning PySpark_, Chapter 11: Packing Spark Applications, Packt, 2017

*Lab*: **Click prediction using 1TB Criteo dataset**
* Load Criteo [dataset](http://labs.criteo.com/2015/03/criteo-releases-its-new-dataset/) into RDD 
* Deploy a MapReduce job to compute the click through rate in a cluster system with varying number of machines
* Training a machine learning algorithm for click prediction using the Criteo dataset 
