Indexing, Slicing, Joining, Mutating Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cat:

cat
^^^

term (up to a line of text)
   Definition of the term, which must be indented

   and can even consist of multiple paragraphs

next term
   Description.

| These lines are
| broken exactly like in
| the source file.

This is a normal text paragraph. The next paragraph is a code sample::

   It is not processed in any way, except
   that the indentation is removed.

   It can span multiple lines.

This is a normal text paragraph again.

.. topic:: Topic Title

    Subsequent indented lines comprise
    the body of the topic, and are
    interpreted as body elements.

.. sidebar:: Optional Sidebar Title
   :subtitle: Optional Sidebar Subtitle

   Subsequent indented lines comprise
   the body of the sidebar, and are
   interpreted as body elements.

.. code:: python

  def my_function():
      "just a test"
      print 8/2

Example:

    >>> x = echotorch.randn(2, length=100)
    >>> x.size()
    torch.Size([2, 100])
    >>> y = echotorch.randn(3, length=100)
    torch.Size([3, 100])
    >>> z = torch.cat((x, y), dim=0)
    >>> z.size()
    torch.Size([5, 100])
    >>> type(z)
    echotorch.timetensors.TimeTensor
    >>> z.tlen
    100
    >>> z.bsize()
    torch.Size([2])
    >>> z.time_dim
    0