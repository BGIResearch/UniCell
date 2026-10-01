.. UniCell documentation master file

Welcome to UniCell's Documentation!
====================================

**UniCell** is a hierarchical multi-task deep learning framework that integrates Cell Ontology with transcriptomic features to enable accurate, scalable, and cross-species annotation and harmonization in single-cell transcriptomics. The current model jointly learns ontology-guided cell types, tissues, and organisms, and can use organism-to-tissue-to-cell-type constraints during inference.

The implementation names the organism task ``species``: the default input column is ``adata.obs["organism"]``, while predictions are stored in ``adata.obs["predicted_species"]``.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: 📘 User Guide

   TUTORIALS/index

.. toctree::
   :maxdepth: 2
   :caption: 🧬 API Reference

   API/index

.. toctree::
   :maxdepth: 2
   :caption: 📦 Package Structure

   PACKAGE ORGANIZATION/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
