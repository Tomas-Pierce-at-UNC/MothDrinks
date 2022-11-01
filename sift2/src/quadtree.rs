use crate::keypoint::Descriptor;

use rand::seq::SliceRandom;


use ordered_float::NotNan;
pub use bheap::DescPriorityQueue;

pub struct Node {
	value :Descriptor,
	northeast :Option<Box<Node>>,
	northwest :Option<Box<Node>>,
	southeast :Option<Box<Node>>,
	southwest :Option<Box<Node>>,
}

impl Node {
	fn new(value :Descriptor) -> Node {
		Node {
			value,
			northeast : None,
			northwest : None,
			southeast : None,
			southwest : None,
		}
	}

	fn insert_node(&mut self, node :Node) {
		let x = node.value.get_x();
		let y = node.value.get_y();
		let x_thresh = self.value.get_x();
		let y_thresh = self.value.get_y();
		if x <= x_thresh && y <= y_thresh {
			// northwest case
			match &mut self.northwest {
				None => {
					self.northwest = Some(Box::new(node));
				},
				Some(nw) => {
					nw.insert_node(node);
				}
			};
		} else if x > x_thresh && y <= y_thresh {
			// northeast case
			match &mut self.northeast {
				None => {
					self.northeast = Some(Box::new(node));
				},
				Some(ne) => {
					ne.insert_node(node);
				}
			}
		} else if x <= x_thresh && y > y_thresh {
			// southwest case
			match &mut self.southwest {
				None => {
					self.southwest = Some(Box::new(node));
				},
				Some(sw) => {
					sw.insert_node(node);
				},
			};
		} else {
			// south east case
			match &mut self.southeast {
				None => {
					self.southeast = Some(Box::new(node));
				},
				Some(se) => {
					se.insert_node(node);
				}
			}
		}
	}

	pub fn insert(&mut self, item :Descriptor) {
		let new_node = Node::new(item);
		self.insert_node(new_node);
	}

	pub fn build(mut descriptors :Vec<Descriptor>) -> Node {
		let des = &mut descriptors;
		let mut rng = rand::thread_rng();
		des.shuffle(&mut rng);
		let start = &descriptors.pop().unwrap();
		let mut root = Node::new(start.clone());
		for item in descriptors {
			root.insert(item);
		}
		root
	}

	pub fn find_closest(&self, descriptor :&Descriptor) -> Descriptor {
		let curr_dist = self.value.distance_squared(descriptor);
		let nw_dist = match &self.northwest {
			Some(nw) => nw.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let ne_dist = match &self.northeast {
			Some(ne) => ne.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let se_dist = match &self.southeast {
			Some(se) => se.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let sw_dist = match &self.southwest {
			Some(sw) => sw.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let min = curr_dist.min(nw_dist).min(ne_dist).min(se_dist).min(sw_dist);
		if min == curr_dist {
			self.value.clone()
		} else if min == ne_dist {
			self.northeast.as_ref().unwrap().find_closest(descriptor)
		} else if min == nw_dist {
			self.northwest.as_ref().unwrap().find_closest(descriptor)
		} else if min == sw_dist {
			self.southwest.as_ref().unwrap().find_closest(descriptor)
		} else {
			self.southeast.as_ref().unwrap().find_closest(descriptor)
		}
	}

}

impl Node {
		fn find_canidates(&self, descriptor :&Descriptor, canidates :&mut DescPriorityQueue) {
		let curr_dist = self.value.distance_squared(descriptor);
		let nw_dist = match &self.northwest {
			Some(nw) => nw.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let ne_dist = match &self.northeast {
			Some(ne) => ne.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let se_dist = match &self.southeast {
			Some(se) => se.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let sw_dist = match &self.southwest {
			Some(sw) => sw.value.distance_squared(descriptor),
			None => NotNan::new(f64::INFINITY).unwrap(),
		};
		let min = curr_dist.min(nw_dist).min(ne_dist).min(se_dist).min(sw_dist);
		if min == curr_dist {
			canidates.push(&self.value)
		} else if min == ne_dist {
			let ne = self.northeast.as_ref().unwrap();
			canidates.push(&Box::as_ref(ne).value);
			ne.find_canidates(descriptor, canidates);
		} else if min == nw_dist {
			let nw = self.northwest.as_ref().unwrap();
			canidates.push(&Box::as_ref(nw).value);
			nw.find_canidates(descriptor, canidates);
		} else if min == sw_dist {
			let sw = self.southwest.as_ref().unwrap();
			canidates.push(&Box::as_ref(sw).value);
			sw.find_canidates(descriptor, canidates);
		} else {
			let se = self.southeast.as_ref().unwrap();
			canidates.push(&Box::as_ref(se).value);
			se.find_canidates(descriptor, canidates);
		}
	}

	pub fn find_closest_approx_multiple(&self, descriptor :&Descriptor) -> DescPriorityQueue {
		let mut pq = DescPriorityQueue::new(descriptor.clone());
		self.find_canidates(descriptor, &mut pq);
		pq
	}
}

mod bheap {
	use ordered_float::NotNan;
	use crate::keypoint::Descriptor;
	use std::cmp::Ordering;
	use binary_heap_plus::BinaryHeap;
	use binary_heap_plus::MinComparator;

	type MinHeap<T> = BinaryHeap<T, MinComparator>;

	pub struct Item {
		distance :NotNan<f64>,
		desc :Descriptor,
	}

	impl Item {
		pub fn new(item :Descriptor, reference :&Descriptor) -> Item {
			let dist = reference.distance_squared(&item);
			Item {
				distance : dist,
				desc : item.clone(),
			}
		}
	}

	impl PartialEq for Item {
		fn eq(&self, other :&Item) -> bool {
			self.distance == other.distance
		}
	}

	impl PartialOrd for Item {
		fn partial_cmp(&self, other :&Item) -> Option<Ordering> {
			if self.distance < other.distance {
				Some(Ordering::Less)
			} else if self.distance == other.distance {
				Some(Ordering::Equal)
			} else {
				Some(Ordering::Greater)
			}
		}
	}

	impl Eq for Item {}

	impl Ord for Item {
		fn cmp(&self, other :&Item) -> Ordering {
			if self.distance < other.distance {
				Ordering::Less
			} else if self.distance == other.distance {
				Ordering::Equal
			} else {
				Ordering::Greater
			}
		}
	}

	pub struct DescPriorityQueue {
		reference :Descriptor,
		heap :MinHeap<Item>,
	}

	impl DescPriorityQueue {
		pub fn new(reference :Descriptor) -> DescPriorityQueue {
			DescPriorityQueue {
				reference : reference,
				heap : MinHeap::new_min()
			}
		}

		pub fn push(&mut self, desc :&Descriptor) {
			let item = Item::new(desc.clone(), &self.reference);
			self.heap.push(item);
		}

		pub fn peak(&self) -> Option<&Descriptor> {
			match self.heap.peek() {
				Some(item) => Some(&item.desc),
				None => None
			}
		}

		pub fn pop(&mut self) -> Option<Descriptor> {
			match self.heap.pop() {
				Some(item) => Some(item.desc),
				None => None
			}
		}

		pub fn is_empty(&self) -> bool {
			self.heap.is_empty()
		}
	}
}