'use strict';

var app = angular.module('QSIApp', ['elasticsearch','angular-advanced-searchbox','ngMaterial', 'smart-table']); 
/*

app.service('client', function (esFactory) {
	  return esFactory({ host: 'localhost:9200/'});
	  
	});
*/

app.factory("trie", function(){	
	var trie = []; 
	trie.selectbynum = function( dict){
		
		var keys = Object.keys(dict) ; 
		
		var kk = [] ; 
		var dd = [] ; 
		
		for (var k in keys) {
			//if (Number.isInteger((k)) {
				kk.push(k);
			//};
		}
		//kk.sort(); 
		for (var k2  in kk ){
			if (dict[k2]  ) {
					 dd.push(dict[k2]);
			}
		}	
		return dd; 
	};
	return trie ; 
} );

app.factory('elastic_word', ["$http",  function ($http ){
	var elastic = {};
	elastic.search = function (query, domain ) {    
		//return $http.get("localhost:9200/my_index/_search?q=_all:"+ val)
		return $http(
				{
			
				method: "POST" ,
				url: "http://35.167.29.71:5001/words",
				//url : "http://127.0.0.1:9200/quran-index/ayats/_search?q="+"الفِيل" ,
				//params :{"query":{"match_all":{}}},
				data:  { query , "domain" : domain },// {"query":{"term":{"السورة":"الفِيل"}}},
				headers : {'Content-Type' : 'application/x-www-form-urlencoded' }
				});
		};
	return elastic  ;
}]);

app.factory('elastic_ayat', ["$http",  function ($http ){
	var elastic = {};
	elastic.search = function (query, domain ) {    
		//return $http.get("localhost:9200/my_index/_search?q=_all:"+ val)
		return $http(
				{
			
				method: "POST" ,
				url: "http://35.167.29.71:5001/ayats",
				//url : "http://127.0.0.1:9200/quran-index/ayats/_search?q="+"الفِيل" ,
				//params :{"query":{"match_all":{}}},
				data:  { query , "domain" : domain },// {"query":{"term":{"السورة":"الفِيل"}}},
				headers : {'Content-Type' : 'application/x-www-form-urlencoded' }
				});
		};
	return elastic  ;
}]);



app.controller('MainCtrl',  function($scope, $http, elastic_word, $rootScope, $timeout)  {
	
		$scope.availableSearchParams = [];/*
          { key: "السورة", name: "السورة", placeholder: "السورة" },
          { key: "مكان النزول", name: "مكان النزول", placeholder: "مكان النزول" },
          { key: "ترتيب السورة", name: "ترتيب السورة", placeholder: "ترتيب السورة" },
         // { key: "الآية", name: "الآية", placeholder: "الآية", allowMultiple: true },
          
        ];*/
		
		$rootScope.domain = "بحث عام";
		$scope.radioData = [
		      { label: 'بحث عام', value: "بحث عام" },
		      { label: 'الكلمات المتقاربة', value: "الكلمات المتقاربة" },
		     // { label: '3', value: "3", isDisabled: true },
		     // { label: '4', value: "4", isDisabled: true }
		 ];
		
		
		
		$rootScope.rest =[]; 
	    
	    
	 	$scope.checkIfEnterKeyWasPressed = function f1 ($event ){
		    var keyCode = $event.which || $event.keyCode;
		    if (keyCode === 13 ) {
		    	
		    	 $timeout(function(){ 
		    	
		    	if (Object.keys($scope.searchParams).length !== 0 ) 
			 		
		 		{

		    	elastic_word.search ($scope.searchParams, $rootScope.domain)
		    	.then(function (data) {
		
		   		$rootScope.rest  = data.data

		 	               
	            	})
	            	.error(function (error) {
	            		$scope.error = error; 
	           	 });
		    	
		 	}
		    	}, 2000);
			 
		    }
	 	};
	 	
	
	 	 
	 	  
	 	  
	 	$scope.domainchange = function f2( dom) {
	 		$rootScope.domain = dom;
	 		$rootScope.rest2 = {};
 			$rootScope.selected = [];
	 		$timeout(function(){ 
	 		
	 		if (Object.keys($scope.searchParams).length !== 0 ) 
			{
	 			
	 		elastic_word.search($scope.searchParams, dom)
	 		.then(function (data) {
	 			$rootScope.rest  = data.data  ; 
            		})
            		.error(function (error) {
            			$scope.error = error ;
            		});
	 		}
	 		}, 2000);
	 		
	 	};
	 	
	 	
	 	  
	 	$scope.$watch('searchParams', function() {
	 		
	 		if (Object.keys($scope.searchParams).length === 0 ) 
	 		
	 		{
	 			$rootScope.rest  = []; 
	 			$rootScope.rest2 = {};
	 			$rootScope.selected = [];
	 		}
	 		
	 	});



});

app.controller('ResultCtrl', function($scope, $rootScope, elastic_ayat, trie, $location) {
	$rootScope.rest2 = {};
	$rootScope.selected = [];
	$scope.show = false ;  
	$rootScope.$watchCollection('rest', function() {
		$rootScope.selected = [];
		if (Object.keys($rootScope.rest).length !== 0 ) {
			$scope.show = true ; 
		
		} else {
		$scope.show = false ;
		$rootScope.rest2  = {};
		$rootScope.selected = [];
		}
	},true);
	
	 
	  
	  $scope.toggle = function (item, list) {
	    var idx = list.indexOf(item);
	    if (idx > -1) {
	      list.splice(idx, 1);
	    }
	    else {
	      list.push(item);
	    }
	  };

	  $scope.exists = function (item, list) {
	    return list.indexOf(item) > -1;
	  };

	  $scope.isIndeterminate = function() {
	    return ($rootScope.selected.length !== 0 &&
	        $rootScope.selected.length !== $rootScope.rest.length);
	  };

	  $scope.isChecked = function() {
	    return $rootScope.selected.length === $rootScope.rest.length;
	  };

	  $scope.toggleAll = function() {
	    if ($rootScope.selected.length === $rootScope.rest.length) {
	      $rootScope.selected = [];
	    } else if ($rootScope.selected.length === 0 || $rootScope.selected.length > 0) {
	      $rootScope.selected = $rootScope.rest.slice(0);
	    }
	  };
		 
	
	$rootScope.show2 = false;
	$rootScope.$watchCollection('selected', function() {
 		
		
 		if (Object.keys($rootScope.selected).length !== 0 ) 
 		
 		{
 			$rootScope.show2 = true;
 		elastic_ayat.search($rootScope.selected, $rootScope.domain)
 		.then(function (data) {
 			$rootScope.rest2  = data.data  ; 
        })
        .error(function (error) {
        	$scope.error2 = error ;
        });
 		}else {
 			$rootScope.show2 = false;
 			$rootScope.rest2  = {};
 		}
 	},true); 
	
	$rootScope.show3 = false ;  
	$rootScope.$watchCollection('rest2', function() {
		
		if (Object.keys($rootScope.rest2).length !== 0 ) {
			$rootScope.show3 = true 	;}
		else {
			$rootScope.show3 = false;}
		
		
		
	},true); 
	
	
	$scope.go = function ( path, show  ) {
		if (show == true ){
		  $location.path( path );
		}
		return show;
		};
	
	
	$scope.select = trie ; 
	// $rootScope.rest2.list_surats["الآية"].sort();
	

	
});

app.controller('SuratCtrl', function($scope, $rootScope, elastic_ayat) {

});
